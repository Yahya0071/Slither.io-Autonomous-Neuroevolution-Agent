/**
 * Persistence & Cross-Origin Checkpointing v5.0.0 Unit Tests
 *
 * Verifies:
 * 1. Generation is incremented before saving or saved accurately so reloading a Gen 1
 *    completion starts at generation === 2 (not stuck at 1).
 * 2. Full checkpoint state (kills, teacherSamples, currentEpisode, episodeFitnesses,
 *    candidatePending, activeBrain, championBrain, allTimePeakLength) round-trips
 *    through localStorage across agent restarts.
 * 3. A Teacher Mode session that dies after >= 15s and >= 300 samples persists
 *    candidatePending = true and activeBrain to localStorage immediately on death,
 *    and survives a simulated browser restart.
 * 4. Mid-generation rollout checkpointing (Episode 1 and 2 persist progress).
 * 5. Bridge state adoption rules (higher fitness, higher gen, or newer timestamp).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadAgent, fingerprint, sameGenome, MemoryStorage, sourceCode } from '../agent-loader.mjs';

const INPUTS = 32;

test('generation is incremented before saving so reloading a Gen 1 completion starts at generation === 2', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();

  assert.equal(agent.generation, 1, 'starts at generation 1');
  assert.equal(agent.currentEpisode, 1, 'starts at episode 1');

  // Complete a full 3-episode rollout for Generation 1
  agent.fitness = 100;
  agent.onDeath(false); // Episode 1 finishes -> currentEpisode becomes 2
  assert.equal(agent.currentEpisode, 2);

  agent.fitness = 200;
  agent.onDeath(false); // Episode 2 finishes -> currentEpisode becomes 3
  assert.equal(agent.currentEpisode, 3);

  agent.fitness = 300;
  agent.onDeath(false); // Episode 3 finishes -> Gen 1 complete, advances to Gen 2

  assert.equal(agent.generation, 2, 'in-memory agent advances to generation 2');
  assert.equal(agent.currentEpisode, 1, 'in-memory agent resets to episode 1');

  // Verify what was written to storage
  const serialized = storage.getItem(harness.STORAGE_KEY);
  assert.ok(serialized, 'checkpoint must be saved to storage');
  const parsed = JSON.parse(serialized);
  assert.equal(parsed.generation, 2, 'saved checkpoint must have generation 2, not 1');
  assert.equal(parsed.currentEpisode, 1);
  assert.deepEqual(parsed.episodeFitnesses, []);

  // Simulate browser restart by loading a fresh agent from the same storage
  const resumed = harness.newAgent();
  assert.equal(resumed.generation, 2, 'reloaded agent starts at generation 2 (not stuck at 1)');
  assert.equal(resumed.currentEpisode, 1, 'reloaded agent starts at episode 1 for Gen 2');
});

test('full checkpoint state round-trips through localStorage across agent restarts', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();

  // Create a distinct champion brain
  agent.championBrain = harness.NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);
  agent.championBrain.weights[0][0] = 3.14159;

  // Create a distinct active brain
  agent.brain = harness.NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);
  agent.brain.weights[0][0] = -2.71828;

  // Set all full state fields
  agent.bestFitness = 1450.75;
  agent.allTimePeakFitness = 1800.5;
  agent.generation = 4;
  agent.currentEpisode = 2;
  agent.episodeFitnesses = [320];
  agent.candidatePending = true;
  agent.candidateSource = 'teacher';
  agent.kills = 9;
  agent.teacherSamples = 650;
  agent.allTimePeakLength = 1280;
  agent.stagnantGens = 2;

  // Persist full state
  agent._saveChampionWeights();

  const rawSaved = storage.getItem(harness.STORAGE_KEY);
  assert.ok(rawSaved);
  const parsed = JSON.parse(rawSaved);

  assert.equal(parsed.bestFitness, 1450.75);
  assert.equal(parsed.allTimePeakFitness, 1800.5);
  assert.equal(parsed.generation, 4);
  assert.equal(parsed.currentEpisode, 2);
  assert.deepEqual(parsed.episodeFitnesses, [320]);
  assert.equal(parsed.candidatePending, true);
  assert.equal(parsed.candidateSource, 'teacher');
  assert.equal(parsed.kills, 9);
  assert.equal(parsed.teacherSamples, 650);
  assert.equal(parsed.allTimePeakLength, 1280);
  assert.equal(parsed.stagnantGens, 2);
  assert.ok(parsed.brain, 'champion brain export present');
  assert.ok(parsed.activeBrain, 'active brain export present');

  // Restart agent from storage
  const restarted = harness.newAgent();
  assert.equal(restarted.bestFitness, 1450.75);
  assert.equal(restarted.allTimePeakFitness, 1800.5);
  assert.equal(restarted.generation, 4);
  assert.equal(restarted.currentEpisode, 2);
  assert.deepEqual(restarted.episodeFitnesses, [320]);
  assert.equal(restarted.candidatePending, true);
  assert.equal(restarted.candidateSource, 'teacher');
  assert.equal(restarted.kills, 9);
  assert.equal(restarted.teacherSamples, 650);
  assert.equal(restarted.allTimePeakLength, 1280);
  assert.equal(restarted.stagnantGens, 2);

  // Both genomes preserved accurately
  assert.ok(sameGenome(fingerprint(restarted.championBrain), fingerprint(agent.championBrain)));
  assert.ok(sameGenome(fingerprint(restarted.brain), fingerprint(agent.brain)));

  // Test predictions match
  const testInputs = new Float32Array(INPUTS).fill(0.3);
  const expectedPred = Array.from(agent.brain.predict(testInputs));
  const actualPred = Array.from(restarted.brain.predict(testInputs));
  assert.deepEqual(actualPred, expectedPred);
});

test('a Teacher Mode session that dies after >= 15s and >= 300 samples persists candidatePending = true and activeBrain to localStorage immediately on death, and survives a simulated browser restart', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();
  agent.bestFitness = 800;
  agent.championBrain = harness.NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);
  const originalChampion = fingerprint(agent.championBrain);

  // Teacher mode session
  agent.brain.mutate(0.4, 0.2); // weight adjustments from tuition
  const teacherGenome = fingerprint(agent.brain);
  agent.touchedByHuman = true;
  agent.teacherSamples = 450; // >= 300
  agent.survivalTime = 25;    // >= 15s
  agent.fitness = 75;

  // Snake dies while human was playing
  agent.onDeath(true);

  // Assert candidate is queued in memory
  assert.equal(agent.candidatePending, true);
  assert.equal(agent.candidateSource, 'teacher');
  assert.equal(agent.currentEpisode, 1);
  assert.deepEqual(agent.episodeFitnesses, []);

  // Assert immediately saved to localStorage
  const savedRaw = storage.getItem(harness.STORAGE_KEY);
  assert.ok(savedRaw, 'checkpoint must be written immediately on candidate queue');
  const saved = JSON.parse(savedRaw);
  assert.equal(saved.candidatePending, true);
  assert.equal(saved.candidateSource, 'teacher');
  assert.equal(saved.teacherSamples, 450);
  assert.ok(sameGenome(fingerprint(harness.NeuralNetwork.fromJSON(saved.brain, INPUTS)), originalChampion), 'champion weights preserved in storage');
  assert.ok(sameGenome(fingerprint(harness.NeuralNetwork.fromJSON(saved.activeBrain, INPUTS)), teacherGenome), 'trained teacher brain saved as activeBrain in storage');

  // Simulate browser close / restart
  const restarted = harness.newAgent();
  assert.equal(restarted.candidatePending, true, 'candidatePending restored');
  assert.equal(restarted.candidateSource, 'teacher', 'candidateSource restored');
  assert.equal(restarted.teacherSamples, 450, 'teacher lessons restored');
  assert.equal(restarted.currentEpisode, 1, 'candidate evaluation starts on episode 1');
  assert.deepEqual(restarted.episodeFitnesses, []);
  assert.equal(restarted.bestFitness, 800, 'champion fitness bar preserved');
  assert.ok(sameGenome(fingerprint(restarted.championBrain), originalChampion), 'champion genome preserved');
  assert.ok(sameGenome(fingerprint(restarted.brain), teacherGenome), 'active brain is the teacher candidate');
});

test('mid-generation rollout episodes (episodes 1 and 2) persist to localStorage immediately', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();

  agent.fitness = 400;
  agent.onDeath(false); // Episode 1 finishes

  const snap1 = JSON.parse(storage.getItem(harness.STORAGE_KEY));
  assert.equal(snap1.currentEpisode, 2, 'storage reflects episode 2');
  assert.deepEqual(snap1.episodeFitnesses, [400], 'storage has episode 1 fitness');

  // Restart mid-generation
  const midGen = harness.newAgent();
  assert.equal(midGen.currentEpisode, 2);
  assert.deepEqual(midGen.episodeFitnesses, [400]);

  // Complete episode 2
  midGen.fitness = 600;
  midGen.onDeath(false);

  const snap2 = JSON.parse(storage.getItem(harness.STORAGE_KEY));
  assert.equal(snap2.currentEpisode, 3);
  assert.deepEqual(snap2.episodeFitnesses, [400, 600]);
});

test('applySavedState accepts valid payloads and rejects malformed ones', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();

  assert.equal(agent.applySavedState(null), false);
  assert.equal(agent.applySavedState(undefined), false);
  assert.equal(agent.applySavedState({}), false);
  assert.equal(agent.applySavedState({ brain: null }), false);

  const validExport = agent.brain.export();
  const success = agent.applySavedState({
    brain: validExport,
    bestFitness: 999,
    generation: 5,
    currentEpisode: 3,
    kills: 12,
    teacherSamples: 340,
    allTimePeakLength: 750,
    stagnantGens: 1,
    timestamp: 123456789
  });
  assert.equal(success, true);
  assert.equal(agent.bestFitness, 999);
  assert.equal(agent.generation, 5);
  assert.equal(agent.currentEpisode, 3);
  assert.equal(agent.kills, 12);
  assert.equal(agent.teacherSamples, 340);
  assert.equal(agent.allTimePeakLength, 750);
  assert.equal(agent.stagnantGens, 1);
  assert.equal(agent.lastLoadedTimestamp, 123456789);
});

test('reloading a Gen 1 checkpoint after Episode 1 (bestFitness === 0, hasChampion === false, episodeFitnesses = [420]) keeps championBrain === null while restoring episodeFitnesses and currentEpisode === 2', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();

  agent.fitness = 420;
  agent.onDeath(false); // Episode 1 finishes -> episodeFitnesses = [420], currentEpisode = 2

  const saved = JSON.parse(storage.getItem(harness.STORAGE_KEY));
  assert.equal(saved.currentEpisode, 2);
  assert.deepEqual(saved.episodeFitnesses, [420]);
  assert.equal(saved.bestFitness, 0);
  assert.equal(saved.hasChampion, false);

  const reloaded = harness.newAgent();
  assert.equal(reloaded.championBrain, null, 'championBrain must remain null until a champion is crowned');
  assert.equal(reloaded.currentEpisode, 2);
  assert.deepEqual(reloaded.episodeFitnesses, [420]);
  assert.equal(reloaded.bestFitness, 0);
});

test('applySavedState restores championBrain when hasChampion === true or bestFitness > 0', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();
  const exportNet = agent.brain.export();

  // 1. hasChampion === true with bestFitness === 0
  const ok1 = agent.applySavedState({
    brain: exportNet,
    hasChampion: true,
    bestFitness: 0,
    generation: 1
  });
  assert.equal(ok1, true);
  assert.notEqual(agent.championBrain, null, 'championBrain restored when hasChampion === true');

  // 2. bestFitness > 0 with hasChampion === false
  const ok2 = agent.applySavedState({
    brain: exportNet,
    hasChampion: false,
    bestFitness: 750,
    generation: 2
  });
  assert.equal(ok2, true);
  assert.notEqual(agent.championBrain, null, 'championBrain restored when bestFitness > 0');

  // 3. hasChampion === false and bestFitness === 0 -> championBrain remains null
  const ok3 = agent.applySavedState({
    brain: exportNet,
    hasChampion: false,
    bestFitness: 0,
    generation: 1
  });
  assert.equal(ok3, true);
  assert.equal(agent.championBrain, null, 'championBrain is null when hasChampion is false and bestFitness is 0');
});

test('resetBrain clears storage and invokes __slitherMLBridgeClear when present', () => {
  let bridgeClearCalled = false;
  const mockWin = {
    __slitherMLBridgeClear() {
      bridgeClearCalled = true;
    }
  };
  const storage = new MemoryStorage();
  storage.setItem('slither_ml_champion_weights_v4', '{"generation":3}');
  const harness = loadAgent({ storage, window: mockWin });
  const agent = harness.newAgent();
  agent.resetBrain();

  assert.equal(bridgeClearCalled, true, 'window.__slitherMLBridgeClear must be invoked on reset');
  assert.equal(storage.getItem('slither_ml_champion_weights_v4'), null, 'localStorage item must be cleared');
  assert.equal(agent.generation, 1);
  assert.equal(agent.championBrain, null);
});

test('loadApexPreset restores the Gen 100 Champion baseline with 37.1k bestFitness and breeds exploration mutant', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();
  agent.resetBrain();

  assert.equal(agent.generation, 1);
  assert.equal(agent.bestFitness, 0);

  const loaded = agent.loadApexPreset();
  assert.equal(loaded, true);
  assert.equal(agent.generation, 100);
  assert.equal(agent.bestFitness, 37089);
  assert.equal(agent.allTimePeakFitness, 43634);
  assert.equal(agent.allTimePeakLength, 22783);
  assert.equal(agent.kills, 168);
  assert.equal(agent.teacherSamples, 1079717);
  assert.notEqual(agent.championBrain, null);
  assert.equal(agent.currentEpisode, 1);
  assert.deepEqual(agent.episodeFitnesses, []);

  // Forward pass works accurately on both champion and exploration mutant
  const testInputs = new Float32Array(32).fill(0.2);
  const champOut = agent.championBrain.predict(testInputs);
  assert.equal(champOut.length, 2);
  assert.ok(champOut[0] >= -1 && champOut[0] <= 1, 'champion steer bounded');
  assert.ok(champOut[1] >= 0 && champOut[1] <= 1, 'champion boost bounded');

  // Verify Gen 100 steering bias is centered
  assert.ok(Math.abs(agent.championBrain.biases[2][0]) < 0.01, 'Gen 100 steering bias is centered');

  const out = agent.brain.predict(testInputs);
  assert.equal(out.length, 2);
  assert.ok(out[0] >= -1 && out[0] <= 1, 'mutant steer bounded');
  assert.ok(out[1] >= 0 && out[1] <= 1, 'mutant boost bounded');
});

test('getApexPreset decodes valid 32-48-24-2 weights matching Gen 100 Champion', () => {
  const harness = loadAgent();
  const preset = harness.SlitherMLAgent.getApexPreset();
  assert.ok(preset, 'preset must exist');
  assert.equal(preset.bestFitness, 37089);
  assert.equal(preset.allTimePeakFitness, 43634);
  assert.equal(preset.generation, 100);
  assert.equal(preset.kills, 168);
  assert.equal(preset.teacherSamples, 1079717);
  assert.equal(preset.brain.inputNodes, 32);
  assert.deepEqual(preset.brain.hiddenLayers, [48, 24]);
  assert.equal(preset.brain.outputNodes, 2);
  assert.equal(preset.brain.weights[0].length, 32 * 48);
  assert.equal(preset.brain.weights[1].length, 48 * 24);
  assert.equal(preset.brain.weights[2].length, 24 * 2);
  assert.equal(preset.brain.biases[0].length, 48);
  assert.equal(preset.brain.biases[1].length, 24);
  assert.equal(preset.brain.biases[2].length, 2);
  assert.ok(Math.abs(preset.brain.biases[2][0]) < 0.01, 'steering output bias is neutral');
});

test('allTimePeakFitness survives page reload and strictly enforces the 85% plateau decay floor across restarts', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();

  // Peak reached 50,000, then stagnated for 10 generations with decayed bestFitness
  agent.bestFitness = 45000;
  agent.allTimePeakFitness = 50000;
  agent.generation = 15;
  agent.stagnantGens = 10;
  agent.championBrain = harness.NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);

  // Save state
  agent._saveChampionWeights();

  // Reload new agent from same storage (simulating page refresh)
  const reloaded = harness.newAgent();
  assert.equal(reloaded.bestFitness, 45000);
  assert.equal(reloaded.allTimePeakFitness, 50000, 'peak fitness must survive page reload');

  // Trigger further stagnation that attempts to decay past the 85% floor (85% of 50,000 = 42,500)
  reloaded.bestFitness = 42600;
  reloaded.stagnantGens = 15; // 1.0% decay would give 42,174, but floor is 42,500
  reloaded.currentEpisode = 3;
  reloaded.episodeFitnesses = [1000, 1000];
  reloaded.fitness = 1000;
  reloaded.onDeath(false); // triggers _recordEvaluationEpisode plateau decay

  assert.equal(reloaded.bestFitness, 42500, 'plateau relief must strictly respect the 85% floor derived from persistent allTimePeakFitness');
});

test('SlitherMLController.loadApexPreset prompts confirm dialog when active champion has higher fitness', () => {
  let confirmPromptText = '';
  let confirmResult = false;

  const mockWindow = {
    confirm(msg) {
      confirmPromptText = msg;
      return confirmResult;
    }
  };

  const storage = new MemoryStorage();
  const harness = loadAgent({ storage, window: mockWindow });
  const agent = harness.newAgent();
  agent.resetBrain();

  // Give agent a high fitness champion (55,000 > 43,854)
  agent.bestFitness = 55000;
  agent.championBrain = harness.NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);

  const controllerPresetMatch = sourceCode.slice(1600).match(/loadApexPreset\(\) \{\s+try \{\s+const preset = SlitherMLAgent\.getApexPreset\(\);[\s\S]*?\n    \}/);
  assert.ok(controllerPresetMatch, 'controller loadApexPreset must exist in sourceCode');

  const controllerMethod = new Function('SlitherMLAgent', 'pageWindow', `
    return {
      ${controllerPresetMatch[0]}
    }.loadApexPreset;
  `)(harness.SlitherMLAgent, mockWindow);

  let hudUpdated = false;
  const controller = {
    agent,
    _updateHUDMetrics() { hudUpdated = true; },
    loadApexPreset: controllerMethod
  };

  // User declines confirmation
  confirmResult = false;
  const declineOk = controller.loadApexPreset();
  assert.equal(declineOk, false, 'must abort preset load when user cancels confirm dialog');
  assert.ok(confirmPromptText.includes('Active Champion Warning'), 'warning dialog shown to user');
  assert.equal(agent.bestFitness, 55000, 'champion fitness intact');
  assert.equal(hudUpdated, false);

  // User accepts confirmation
  confirmResult = true;
  const acceptOk = controller.loadApexPreset();
  assert.equal(acceptOk, true, 'loads preset when user confirms');
  assert.equal(agent.bestFitness, 37089, 'preset loaded');
  assert.equal(agent.generation, 100);
  assert.equal(hudUpdated, true);
});
