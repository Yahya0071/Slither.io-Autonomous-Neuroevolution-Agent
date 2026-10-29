/**
 * Champion-selection and teacher-mode bookkeeping regression tests.
 *
 * These lock down the four defects found in review:
 *   1. a >=15s teacher run used to overwrite the champion genome outright
 *   2. `bestFitness *= 0.992` used to let strictly worse genomes be crowned
 *   3. one stray [T] frame used to hijack the end-of-life champion save
 *   4. teacher lessons used to be discarded without ever being evaluated
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadAgent, fingerprint, sameGenome, MemoryStorage, sourceCode } from '../agent-loader.mjs';

const INPUTS = 32;
const EPISODES = 3;

/** Agent with an established, strong champion plus a spy on champion saves. */
function withChampion({ fitness = 1000, storage } = {}) {
  const harness = loadAgent({ storage: storage ?? new MemoryStorage(), collectLogs: true });
  const { NeuralNetwork } = harness;
  const agent = harness.newAgent();
  agent.resetBrain();
  agent.bestFitness = fitness;
  agent.championBrain = NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);

  const realSave = agent._saveChampionWeights.bind(agent);
  const saves = { count: 0 };
  agent._saveChampionWeights = () => {
    saves.count++;
    realSave();
  };
  return { ...harness, agent, saves, NeuralNetwork };
}

/** Run one life and report it as a death. */
function die(agent, { fitness, survivalTime = 40, human = false }) {
  agent.fitness = fitness;
  agent.survivalTime = survivalTime;
  agent.onDeath(human);
}

/** Complete a full 3-life rollout at a fixed per-life fitness. */
function rollout(agent, fitness) {
  for (let i = 0; i < EPISODES; i++) die(agent, { fitness });
}

/** Simulate the state the controller leaves after a real teacher session. */
function playTeacherSession(agent, { seconds = 60, lessons = 1200 } = {}) {
  agent.touchedByHuman = true;
  agent.teacherSamples = lessons;
  agent.survivalTime = seconds;
  // stand in for the weight drift that ~3600 backprop steps produce
  agent.brain.mutate(0.5, 0.2);
  agent.fitness = 60;
}

// ---------------------------------------------------------------------------
// 1. short teacher runs are discarded
// ---------------------------------------------------------------------------
test('a teacher life under 15s never touches the champion', () => {
  const { agent, saves, NeuralNetwork } = withChampion();
  const championBefore = fingerprint(NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS));

  playTeacherSession(agent, { seconds: 8, lessons: 900 });
  die(agent, { fitness: 60, survivalTime: 8, human: true });

  assert.equal(saves.count, 0, 'must not write the champion');
  assert.equal(agent.bestFitness, 1000, 'champion fitness unchanged');
  assert.equal(agent.candidatePending, false, 'no candidate queued');
  assert.ok(sameGenome(fingerprint(agent.championBrain), championBefore), 'champion genome preserved');
});

test('a teacher life with too few lessons is discarded even when long', () => {
  const { agent, saves } = withChampion();
  playTeacherSession(agent, { seconds: 60, lessons: 10 });
  die(agent, { fitness: 60, survivalTime: 60, human: true });

  assert.equal(saves.count, 0);
  assert.equal(agent.candidatePending, false);
  assert.equal(agent.bestFitness, 1000);
});

test('an agent loaded with 1.08M cumulative teacherSamples receiving only 10 frames of learnFromHuman is discarded on death', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();
  agent.loadApexPreset(); // sets teacherSamples = 1079717, lastLoadedTimestamp > 0, lifeTeacherSamples = 0
  assert.equal(agent.teacherSamples, 1079717);
  assert.equal(agent.lifeTeacherSamples, 0);

  // Simulate 10 frames of learnFromHuman()
  const snake = { xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, pts: [] };
  for (let i = 0; i < 10; i++) {
    agent.learnFromHuman(snake, [], [], [], 0.016, i * 16, 100, 50, false);
  }
  assert.equal(agent.lifeTeacherSamples, 10);
  assert.equal(agent.teacherSamples, 1079727);

  // Snake dies after 20s of human driving (>= 15s MIN_TEACHER_SECONDS, but lifeTeacherSamples 10 < 300)
  agent.survivalTime = 20;
  agent.fitness = 100;
  agent.onDeath(true);

  assert.equal(agent.candidatePending, false, 'must be discarded because lifeTeacherSamples (10) < 300');
  assert.equal(agent.candidateSource, null);
  assert.equal(agent.lifeTeacherSamples, 0, 'lifeTeacherSamples resets on death');
  assert.equal(agent.teacherSamples, 1079727, 'lifetime teacherSamples is preserved');
});

// ---------------------------------------------------------------------------
// 2. long teacher runs become candidates, not champions
// ---------------------------------------------------------------------------
test('a long teacher life is queued as a candidate and does not write the champion', () => {
  const { agent, saves, NeuralNetwork } = withChampion();
  const championBefore = fingerprint(NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS));

  playTeacherSession(agent);
  die(agent, { fitness: 60, survivalTime: 60, human: true });

  assert.equal(saves.count, 1, 'candidate checkpoint persisted on death');
  assert.equal(agent.bestFitness, 1000, 'the 60-fitness run must not raise or lower the bar');
  assert.equal(agent.candidatePending, true, 'the trained brain is now a candidate');
  assert.equal(agent.candidateSource, 'teacher');
  assert.equal(agent.currentEpisode, 1, 'candidate evaluation starts from episode 1');
  assert.deepEqual(agent.episodeFitnesses, []);
  assert.ok(sameGenome(fingerprint(agent.championBrain), championBefore), 'champion genome untouched');
});

test('a teacher candidate that loses its 3-life evaluation is rejected', () => {
  const { agent, saves, NeuralNetwork, logs } = withChampion();
  const championBefore = fingerprint(NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS));

  playTeacherSession(agent);
  die(agent, { fitness: 60, survivalTime: 60, human: true });
  assert.equal(agent.candidatePending, true);

  rollout(agent, 100); // aggregate 100 << champion 1000

  assert.equal(saves.count, 4, 'candidate queue + 3 episode checkpoints');
  assert.equal(agent.bestFitness, 1000);
  assert.equal(agent.candidatePending, false);
  assert.ok(sameGenome(fingerprint(agent.championBrain), championBefore), 'champion genome preserved');
  assert.ok(logs.some((l) => l.includes('rejected')), 'rejection is reported');
});

test('a teacher candidate that wins its 3-life evaluation becomes the champion', () => {
  const { agent, saves } = withChampion();
  playTeacherSession(agent);
  die(agent, { fitness: 60, survivalTime: 60, human: true });
  const teacherGenome = fingerprint(agent.brain); // the trained brain awaiting evaluation

  rollout(agent, 2000); // aggregate 2000 > champion 1000

  assert.equal(saves.count, 4, 'candidate queue + 3 episode checkpoints');
  assert.equal(agent.bestFitness, 2000, 'bar moves to the evaluated aggregate');
  assert.equal(agent.candidatePending, false);
  assert.ok(sameGenome(fingerprint(agent.championBrain), teacherGenome), 'the winning candidate is the champion');
});

// ---------------------------------------------------------------------------
// 3. the selection bar must not decay
// ---------------------------------------------------------------------------
test('strict elitism holds during first 5 stagnant generations with zero fitness bar decay', () => {
  const { agent, NeuralNetwork } = withChampion({ fitness: 1000 });
  const originalMutate = NeuralNetwork.prototype.mutate;
  NeuralNetwork.prototype.mutate = () => {}; // freeze breeding so every genome is comparable
  try {
    for (let gen = 0; gen < 5; gen++) rollout(agent, 700); // 15 lives at 700
    assert.equal(agent.bestFitness, 1000, `bar must remain strictly at 1000 for 5 generations, got ${agent.bestFitness}`);
    assert.equal(agent.stagnantGens, 5);
  } finally {
    NeuralNetwork.prototype.mutate = originalMutate;
  }
});

test('stagnation relief (bestFitness *= 0.995) only triggers when stagnantGens >= 6 and accelerates to 1.0% on >= 15', () => {
  const { agent } = withChampion({ fitness: 1000 });
  agent.stagnantGens = 0;

  // Gen 1 to 5: bestFitness stays strictly 1000
  for (let g = 1; g <= 5; g++) {
    rollout(agent, 500);
    assert.equal(agent.bestFitness, 1000, `gen ${g} must not decay bar`);
    assert.equal(agent.stagnantGens, g);
  }

  // Gen 6: stagnantGens becomes 6 -> plateau relief activates: 1000 * 0.995 = 995
  rollout(agent, 500);
  assert.equal(agent.stagnantGens, 6);
  assert.equal(Math.round(agent.bestFitness), 995);

  // Gen 7: stagnantGens becomes 7 -> 995 * 0.995 = 990.025
  rollout(agent, 500);
  assert.equal(agent.stagnantGens, 7);
  assert.ok(Math.abs(agent.bestFitness - 990.025) < 1e-3);

  // Fast-forward to stagnantGens = 14
  agent.stagnantGens = 14;
  agent.bestFitness = 800;
  agent.allTimePeakFitness = 800;

  // Gen 15: stagnantGens becomes 15 -> deep plateau relief activates: 800 * 0.990 = 792
  rollout(agent, 500);
  assert.equal(agent.stagnantGens, 15);
  assert.equal(Math.round(agent.bestFitness), 792);
});

test('bestFitness equals the aggregate that produced the champion', () => {
  const { agent } = withChampion({ fitness: 100 });
  rollout(agent, 500);
  assert.equal(agent.bestFitness, 500);

  // a later, worse genome must not move the bar or the genome
  const { agent: agent2 } = { agent };
  rollout(agent2, 300);
  assert.equal(agent2.bestFitness, 500);
});

// ---------------------------------------------------------------------------
// 4. a stray [T] tap must not hijack a life
// ---------------------------------------------------------------------------
test('an autonomous death with a latched human touch does not save the champion', () => {
  const { agent, saves, NeuralNetwork } = withChampion();
  const championBefore = fingerprint(NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS));

  // One frame of teacher mode, then 40s of autopilot, then death with AI driving.
  agent.touchedByHuman = true;
  die(agent, { fitness: 80, survivalTime: 40, human: false });

  assert.equal(saves.count, 0, 'must not take the champion-save path');
  assert.equal(agent.bestFitness, 1000, 'bar unchanged');
  assert.ok(sameGenome(fingerprint(agent.championBrain), championBefore), 'champion genome preserved');
});

test('a partly human-driven life is excluded from the rollout', () => {
  const { agent, saves } = withChampion();
  agent.currentEpisode = 2;
  agent.episodeFitnesses = [400];
  agent.abandonCurrentEpisode = true; // what toggleEnabled() sets when control goes back to the AI

  die(agent, { fitness: 9999 });

  assert.deepEqual(agent.episodeFitnesses, [400], 'the biased life must not be scored');
  assert.equal(agent.currentEpisode, 2, 'episode counter unchanged');
  assert.equal(saves.count, 0);
  assert.equal(agent.abandonCurrentEpisode, false, 'flag clears for the next life');
});

// ---------------------------------------------------------------------------
// 5. telemetry honesty
// ---------------------------------------------------------------------------
test('the generation counter does not inflate on page reload', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const first = harness.newAgent();
  first.resetBrain();
  first.bestFitness = 42;
  first.championBrain = harness.NeuralNetwork.fromJSON(first.brain.export(), INPUTS);
  first._saveChampionWeights();

  let agent = first;
  for (let load = 0; load < 5; load++) agent = harness.newAgent();
  assert.equal(agent.generation, first.generation, 'reloading must not advance the generation');
  assert.equal(agent.bestFitness, 42, 'the saved bar is restored as-is');
});

test('a failed persistence write is reported instead of silently swallowed', () => {
  const storage = new MemoryStorage();
  const { agent, logs } = withChampion({ storage });
  storage.failWrites = true;
  assert.doesNotThrow(() => agent._saveChampionWeights());
  assert.ok(
    logs.some((l) => l.includes('Could not persist')),
    `expected a warning, got: ${JSON.stringify(logs)}`,
  );
});

// ---------------------------------------------------------------------------
// 6. v5.0.3 algorithm enhancements (spawn-kill mulligan & soft teacher blending)
// ---------------------------------------------------------------------------
test('spawn-kill mulligan filter excludes fast low-fitness deaths from rollout', () => {
  const { agent, saves } = withChampion();
  agent.currentEpisode = 1;
  agent.episodeFitnesses = [];

  // Simulate server spawn-kill: died in 1.8s with only 25 fitness (< 6.0s & < 300)
  die(agent, { fitness: 25, survivalTime: 1.8, human: false });

  assert.equal(agent.currentEpisode, 1, 'episode counter must not advance on spawn-kill');
  assert.deepEqual(agent.episodeFitnesses, [], 'spawn-kill must not be recorded');
  assert.equal(saves.count, 0, 'no checkpoint save on excluded mulligan life');

  // Verify second spawn-kill condition: died in 4.2s with 30 fitness and never grew (< 5.0s, < 40 fitness, peakLength <= 10)
  agent.peakLength = 10;
  die(agent, { fitness: 30, survivalTime: 4.2, human: false });
  assert.equal(agent.currentEpisode, 1, 'spawn-kill without food intake must exclude life');
  assert.deepEqual(agent.episodeFitnesses, [], 'spawn-kill must not be recorded');

  // Normal autonomous death: died in 25s with 120 fitness
  die(agent, { fitness: 120, survivalTime: 25, human: false });
  assert.equal(agent.currentEpisode, 2, 'normal life advances episode');
  assert.deepEqual(agent.episodeFitnesses, [120]);
});

test('teacher candidate scoring >= 35% of champion (or >= 5000) blends 20% weights into championBrain without lowering bestFitness and resets stagnantGens', () => {
  const { agent, NeuralNetwork, logs } = withChampion({ fitness: 1000 });
  agent.stagnantGens = 3;
  const championBefore = NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS);

  playTeacherSession(agent);
  die(agent, { fitness: 60, survivalTime: 60, human: true });
  assert.equal(agent.candidatePending, true);

  // Set candidate weights to a distinctive value so we can detect the 20% blend
  const candidateWeights = agent.brain.weights[0];
  candidateWeights.fill(5.0);

  // Freeze mutation during rollout completion so we can inspect the exact blended weights
  const originalMutate = NeuralNetwork.prototype.mutate;
  NeuralNetwork.prototype.mutate = () => {};

  try {
    // Score 400 per episode (aggregate 400 >= 35% of 1000 = 350, but < 1000)
    rollout(agent, 400);

    // 1. Champion fitness bar is maintained without lowering
    assert.equal(agent.bestFitness, 1000, 'champion fitness bar strictly maintained');
    assert.equal(agent.stagnantGens, 0, 'stagnant generations reset to 0 after integrating tuition');

    // 2. Champion brain has blended 20% from candidate (5.0) and 80% from champion
    const expectedFirstWeight = (championBefore.weights[0][0] * 0.8) + (5.0 * 0.2);
    assert.ok(
      Math.abs(agent.championBrain.weights[0][0] - expectedFirstWeight) < 1e-5,
      `expected champion weight ${expectedFirstWeight}, got ${agent.championBrain.weights[0][0]}`
    );

    // 3. Next mutant seed (this.brain) also carries the integrated weights
    assert.ok(
      Math.abs(agent.brain.weights[0][0] - expectedFirstWeight) < 1e-5,
      `expected active weight ${expectedFirstWeight}, got ${agent.brain.weights[0][0]}`
    );
    assert.ok(logs.some((l) => l.includes('Integrating 20% tuition')), 'blend logged');
  } finally {
    NeuralNetwork.prototype.mutate = originalMutate;
  }
});

test('teacher candidate scoring < 35% (and < 5000) does not touch championBrain, but blends 15% weights into mutant breeding template (tuition preservation)', () => {
  const { agent, NeuralNetwork, logs } = withChampion({ fitness: 10000 });
  agent.stagnantGens = 2;
  const championBefore = NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS);

  playTeacherSession(agent);
  die(agent, { fitness: 60, survivalTime: 60, human: true });
  assert.equal(agent.candidatePending, true);

  // Set candidate weights to a distinctive value so we can detect the 15% blend
  const candidateWeights = agent.brain.weights[0];
  candidateWeights.fill(5.0);

  const originalMutate = NeuralNetwork.prototype.mutate;
  NeuralNetwork.prototype.mutate = () => {};

  try {
    // Score 2000 per episode (aggregate 2000 < 35% of 10000 = 3500 and < 5000)
    rollout(agent, 2000);

    // 1. Champion brain and fitness bar must remain 100% untouched
    assert.equal(agent.bestFitness, 10000, 'champion fitness bar strictly preserved');
    assert.equal(agent.championBrain.weights[0][0], championBefore.weights[0][0], 'championBrain remains untouched');
    assert.equal(agent.stagnantGens, 3, 'stagnant generation increments');

    // 2. Breeding template (this.brain) has blended 15% from candidate (5.0) and 85% from champion
    const expectedSeedWeight = (championBefore.weights[0][0] * 0.85) + (5.0 * 0.15);
    assert.ok(
      Math.abs(agent.brain.weights[0][0] - expectedSeedWeight) < 1e-5,
      `expected breeding seed weight ${expectedSeedWeight}, got ${agent.brain.weights[0][0]}`
    );
    assert.ok(logs.some((l) => l.includes('Preserving human tuition')), 'tuition preservation logged');
  } finally {
    NeuralNetwork.prototype.mutate = originalMutate;
  }
});

test('toggleEnabled starts Teacher Mode from pristine championBrain rather than a mutant', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();
  agent.resetBrain();
  agent.bestFitness = 2000;
  agent.championBrain = harness.NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);

  // Perturb agent.brain to simulate an exploration mutant
  agent.brain.weights[0].fill(9.9);

  // Extract and execute the exact toggleEnabled body from sourceCode
  const match = sourceCode.match(/toggleEnabled\(\)\s*\{([\s\S]*?)\n\s*const btn =/);
  assert.ok(match, 'toggleEnabled must be in sourceCode');
  const toggleFn = new Function('NeuralNetwork', match[1]);

  const ctx = { enabled: true, agent };
  toggleFn.call(ctx, harness.NeuralNetwork);

  assert.equal(ctx.enabled, false, 'must switch to Teacher Mode');
  assert.deepEqual(
    agent.brain.weights[0],
    agent.championBrain.weights[0],
    'teacher mode must start from clean champion weights'
  );
});

test('Best-2-of-3 rollout scoring drops lowest life and averages top 2', () => {
  const { agent } = withChampion({ fitness: 43000 });
  agent.currentEpisode = 1;
  agent.episodeFitnesses = [];

  // Simulate 3 lives: life 1 = 26000, life 2 = 73000, life 3 = 1000 (lag spike / cutoff)
  die(agent, { fitness: 26000, survivalTime: 60, human: false });
  assert.equal(agent.currentEpisode, 2);
  assert.deepEqual(agent.episodeFitnesses, [26000]);

  die(agent, { fitness: 73000, survivalTime: 120, human: false });
  assert.equal(agent.currentEpisode, 3);
  assert.deepEqual(agent.episodeFitnesses, [26000, 73000]);

  die(agent, { fitness: 1000, survivalTime: 15, human: false });

  // Evaluation triggered: Best 2 are 73000 and 26000 -> (73000 + 26000) / 2 = 49500
  // Median 26000 >= 43000 * 0.40 (17200) -> passes consistency gate!
  // Since 49500 > 43000, this crowns a new champion!
  assert.equal(agent.bestFitness, 49500, 'Best-2-of-3 aggregate crowns champion over dropped lowest run');
  assert.equal(agent.generation, 2);
  assert.equal(agent.currentEpisode, 1);
  assert.deepEqual(agent.episodeFitnesses, []);
});

test('Consistency Gate rejects 1-run fluke mutants where median run is < 40% of champion', () => {
  const { agent } = withChampion({ fitness: 43000 });
  agent.currentEpisode = 1;
  agent.episodeFitnesses = [];

  // Life 1 = 90000 (fluke), Life 2 = 50 (immediate death), Life 3 = 50 (immediate death)
  // Best-2-of-3 = (90000 + 50) / 2 = 45025 > 43000
  // BUT median (sorted[1]) = 50 < 43000 * 0.40 (17200) -> Consistency Gate rejects promotion!
  die(agent, { fitness: 90000, survivalTime: 80, human: false });
  die(agent, { fitness: 50, survivalTime: 10, human: false });
  die(agent, { fitness: 50, survivalTime: 10, human: false });

  assert.equal(agent.bestFitness, 43000, 'champion bar must not be hijacked by a 1-run fluke');
  assert.equal(agent.stagnantGens, 1, 'counts as stagnant generation');
});

test('autonomous mutant scoring >= 75% of champion blends 15% weights into breeding template without touching championBrain', () => {
  const { agent, NeuralNetwork } = withChampion({ fitness: 50000 });
  const originalMutate = NeuralNetwork.prototype.mutate;
  NeuralNetwork.prototype.mutate = () => {}; // prevent random mutation noise so we can check exact blend math
  try {
    // Fill champion with 1.0 and active mutant with 2.0
    agent.championBrain.weights[0].fill(1.0);
    agent.championBrain.biases[0].fill(1.0);
    agent.brain.weights[0].fill(2.0);
    agent.brain.biases[0].fill(2.0);

    // Life 1 = 45000, Life 2 = 35000, Life 3 = 10000
    // Best-2-of-3 = (45000 + 35000) / 2 = 40000 (40000 / 50000 = 80% >= 75%)
    // Did not win champion (40000 <= 50000), but qualified for 15% mutant soft trait retention!
    die(agent, { fitness: 45000, survivalTime: 80, human: false });
    die(agent, { fitness: 35000, survivalTime: 60, human: false });
    die(agent, { fitness: 10000, survivalTime: 30, human: false });

    // Champion brain must remain strictly untouched at 1.0
    assert.equal(agent.championBrain.weights[0][0], 1.0, 'championBrain must remain untouched');
    assert.equal(agent.bestFitness, 50000, 'bestFitness must remain untouched');

    // Bred next brain should have blended 85% champion (1.0) + 15% mutant (2.0) = 1.15
    const expected = (1.0 * 0.85) + (2.0 * 0.15);
    assert.ok(
      Math.abs(agent.brain.weights[0][0] - expected) < 1e-5,
      `mutant breeding seed must be softly blended to ${expected}, got ${agent.brain.weights[0][0]}`
    );
  } finally {
    NeuralNetwork.prototype.mutate = originalMutate;
  }
});

test('insufficient rollout data (< 2 valid episodes) extends rollout without advancing generation or mutating', () => {
  const { agent } = withChampion({ fitness: 1000 });
  agent.currentEpisode = 1;
  agent.episodeFitnesses = [];
  const startGen = agent.generation;

  // 1 valid life, then mulligans
  die(agent, { fitness: 1200, survivalTime: 30, human: false });
  assert.equal(agent.currentEpisode, 2);
  assert.deepEqual(agent.episodeFitnesses, [1200]);

  // Instant spawn-kills
  die(agent, { fitness: 20, survivalTime: 1.5, human: false });
  die(agent, { fitness: 20, survivalTime: 1.5, human: false });

  assert.equal(agent.generation, startGen, 'generation must not advance on mulligans');
  assert.equal(agent.currentEpisode, 2, 'rollout remains at episode 2 waiting for valid runs');
  assert.equal(agent.bestFitness, 1000, 'cannot crown champion on 1 run');
});

test('plateau decay is strictly capped from below at 85% of all-time peak fitness', () => {
  const { agent } = withChampion({ fitness: 10000 });
  agent.allTimePeakFitness = 10000;
  agent.stagnantGens = 14;

  // Simulate 30 stagnant generations under 1% deep plateau relief
  for (let g = 0; g < 30; g++) {
    rollout(agent, 500);
  }

  // Without floor: 10000 * 0.99^30 = 7397
  // With 85% floor: clamped at exactly 8500
  assert.equal(Math.round(agent.bestFitness), 8500, `bestFitness must be clamped at 85% floor (8500), got ${agent.bestFitness}`);
});
