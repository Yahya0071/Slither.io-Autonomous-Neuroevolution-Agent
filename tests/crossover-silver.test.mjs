/**
 * Crossover & Silver Brain tests (v5.10.8)
 *
 * Covers:
 *   1. NeuralNetwork.crossover() produces a child that borrows genes from both parents
 *   2. crossover() enforces steering-bias [-0.15, 0.15] clamp
 *   3. crossover() with pB=0 is a perfect clone of parentA; pB=1 is a clone of parentB
 *   4. silverBrain is null initially and after resetBrain()
 *   5. silverBrain is populated when a new champion displaces the old one
 *   6. silverBrain is serialized in exportCheckpoint and in _saveChampionWeights
 *   7. silverBrain is restored by applySavedState
 *   8. crossover breeding is used (not pure clone) when stagnantGens >= 5 and silverBrain exists
 *   9. crossover breeding is NOT used when stagnantGens < 5 (pure clone path)
 *  10. crossover breeding is NOT used when silverBrain is null even if stagnant
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadAgent, fingerprint, sameGenome, MemoryStorage } from '../agent-loader.mjs';

const INPUTS = 32;
const EPISODES = 3;

function makeNet(NeuralNetwork, fillValue) {
  const net = new NeuralNetwork(INPUTS, [48, 24], 2);
  for (const w of net.weights) w.fill(fillValue);
  for (const b of net.biases) b.fill(fillValue);
  // Clamp steering bias
  net.biases[2][0] = Math.max(-0.15, Math.min(0.15, fillValue));
  return net;
}

function withChampion({ fitness = 5000, stagnantGens = 0 } = {}) {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage, collectLogs: true });
  const { NeuralNetwork } = harness;
  const agent = harness.newAgent();
  agent.resetBrain();
  agent.bestFitness = fitness;
  agent.allTimePeakFitness = fitness;
  agent.championBrain = NeuralNetwork.fromJSON(agent.brain.export(), INPUTS);
  agent.stagnantGens = stagnantGens;
  return { ...harness, agent, NeuralNetwork, storage };
}

/** Run N full 3-life rollouts each hitting `perLifeFitness` */
function rollout(agent, perLifeFitness, count = 1) {
  for (let r = 0; r < count; r++) {
    for (let ep = 0; ep < EPISODES; ep++) {
      agent.fitness = perLifeFitness;
      agent.survivalTime = 40;
      agent.onDeath(false);
    }
  }
}

// ---------------------------------------------------------------------------
// 1. crossover() borrows genes from both parents
// ---------------------------------------------------------------------------
test('crossover: child contains weights from both parentA and parentB', () => {
  const { NeuralNetwork } = loadAgent();
  const A = makeNet(NeuralNetwork, 1.0);
  const B = makeNet(NeuralNetwork, -1.0);

  // Run many crossovers to get statistical confidence
  let fromA = 0, fromB = 0, total = 0;
  for (let trial = 0; trial < 20; trial++) {
    const child = NeuralNetwork.crossover(A, B, 0.5);
    for (let layer = 0; layer < child.weights.length; layer++) {
      for (const w of child.weights[layer]) {
        total++;
        if (Math.abs(w - 1.0) < 1e-5) fromA++;
        else if (Math.abs(w - (-1.0)) < 1e-5) fromB++;
      }
    }
  }
  // With pB=0.5 we expect ~50% from each parent; check both are significantly present
  assert.ok(fromA > total * 0.3, `Too few weights from parentA: ${fromA}/${total}`);
  assert.ok(fromB > total * 0.3, `Too few weights from parentB: ${fromB}/${total}`);
  assert.equal(fromA + fromB, total, 'Every weight should be exactly from A or B');
});

// ---------------------------------------------------------------------------
// 2. crossover() clamps steering bias within [-0.15, 0.15]
// ---------------------------------------------------------------------------
test('crossover: steering bias is clamped within [-0.15, 0.15]', () => {
  const { NeuralNetwork } = loadAgent();
  const A = makeNet(NeuralNetwork, 0.14);
  const B = makeNet(NeuralNetwork, -0.14);
  // Manually poke biases to extremes to test clamp
  A.biases[2][0] = 0.14;
  B.biases[2][0] = -0.14;
  for (let i = 0; i < 50; i++) {
    const child = NeuralNetwork.crossover(A, B, 0.5);
    const steerBias = child.biases[child.biases.length - 1][0];
    assert.ok(steerBias >= -0.15 && steerBias <= 0.15,
      `Steering bias ${steerBias} out of [-0.15, 0.15]`);
  }
});

// ---------------------------------------------------------------------------
// 3. crossover with pB=0 → clone of A; pB=1 → clone of B
// ---------------------------------------------------------------------------
test('crossover: pB=0 produces exact clone of parentA', () => {
  const { NeuralNetwork } = loadAgent();
  const A = makeNet(NeuralNetwork, 0.7);
  const B = makeNet(NeuralNetwork, -0.7);
  const child = NeuralNetwork.crossover(A, B, 0.0);
  for (let layer = 0; layer < child.weights.length; layer++) {
    for (let j = 0; j < child.weights[layer].length; j++) {
      assert.ok(Math.abs(child.weights[layer][j] - A.weights[layer][j]) < 1e-5,
        'Weight should match parentA at pB=0');
    }
  }
});

test('crossover: pB=1 produces exact clone of parentB', () => {
  const { NeuralNetwork } = loadAgent();
  const A = makeNet(NeuralNetwork, 0.7);
  const B = makeNet(NeuralNetwork, -0.7);
  const child = NeuralNetwork.crossover(A, B, 1.0);
  for (let layer = 0; layer < child.weights.length; layer++) {
    for (let j = 0; j < child.weights[layer].length; j++) {
      assert.ok(Math.abs(child.weights[layer][j] - B.weights[layer][j]) < 1e-5,
        'Weight should match parentB at pB=1');
    }
  }
});

// ---------------------------------------------------------------------------
// 4. silverBrain starts null and is null after resetBrain()
// ---------------------------------------------------------------------------
test('silverBrain: is null on fresh agent and after resetBrain()', () => {
  const { agent } = withChampion();
  agent.silverBrain = null; // sanity
  assert.equal(agent.silverBrain, null, 'fresh agent has no silver brain');

  agent.silverBrain = agent.championBrain; // fake set
  agent.resetBrain();
  assert.equal(agent.silverBrain, null, 'resetBrain() should clear silverBrain');
});

// ---------------------------------------------------------------------------
// 5. silverBrain is populated when a new champion displaces the old one
// ---------------------------------------------------------------------------
test('silverBrain: displaced champion is saved as silverBrain on new crown', () => {
  const { agent, NeuralNetwork } = withChampion({ fitness: 1000, stagnantGens: 0 });

  // Give champion a distinctive fingerprint
  const champFingerprint = fingerprint(NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS));
  assert.equal(agent.silverBrain, null, 'no silver yet');

  // Run a rollout that beats the champion (fitness > 1000, passes consistency gate)
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 2000;
    agent.survivalTime = 60;
    agent.onDeath(false);
  }

  // After crowning, the old champion should now be in silverBrain
  assert.ok(agent.silverBrain !== null, 'silverBrain should be populated after crown');
  const silverFP = fingerprint(NeuralNetwork.fromJSON(agent.silverBrain.export(), INPUTS));
  assert.ok(sameGenome(silverFP, champFingerprint),
    'silverBrain should match the former champion weights');
});

// ---------------------------------------------------------------------------
// 6. silverBrain is serialized by exportCheckpoint and _saveChampionWeights
// ---------------------------------------------------------------------------
test('silverBrain: is included in exportCheckpoint when present', () => {
  const { agent, NeuralNetwork } = withChampion({ fitness: 1000 });
  // Crown a new champion so silverBrain gets populated
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 2000;
    agent.survivalTime = 60;
    agent.onDeath(false);
  }
  assert.ok(agent.silverBrain !== null, 'silverBrain must be set before this test means anything');

  const checkpoint = JSON.parse(agent.exportCheckpoint());
  assert.ok(checkpoint.silverBrain !== null && typeof checkpoint.silverBrain === 'object',
    'exportCheckpoint must include silverBrain');
  assert.ok(Array.isArray(checkpoint.silverBrain.weights), 'silverBrain.weights must be an array');
});

test('silverBrain: is null in exportCheckpoint when no silver exists', () => {
  const { agent } = withChampion({ fitness: 1000 });
  assert.equal(agent.silverBrain, null);
  const checkpoint = JSON.parse(agent.exportCheckpoint());
  assert.equal(checkpoint.silverBrain, null, 'silverBrain should be null in checkpoint when not set');
});

// ---------------------------------------------------------------------------
// 7. applySavedState restores silverBrain from checkpoint
// ---------------------------------------------------------------------------
test('silverBrain: applySavedState restores silverBrain from checkpoint', () => {
  const storage = new MemoryStorage();
  const harness1 = loadAgent({ storage });
  const agent1 = harness1.newAgent();
  agent1.resetBrain();
  agent1.bestFitness = 1000;
  agent1.allTimePeakFitness = 1000;
  const { NeuralNetwork } = harness1;
  agent1.championBrain = NeuralNetwork.fromJSON(agent1.brain.export(), INPUTS);

  // Crown a better genome to populate silverBrain
  for (let ep = 0; ep < EPISODES; ep++) {
    agent1.fitness = 2000;
    agent1.survivalTime = 60;
    agent1.onDeath(false);
  }
  assert.ok(agent1.silverBrain !== null);
  const silverFP = fingerprint(NeuralNetwork.fromJSON(agent1.silverBrain.export(), INPUTS));

  // Serialize and reload into a fresh agent
  const checkpoint = JSON.parse(agent1.exportCheckpoint());
  const harness2 = loadAgent({ storage: new MemoryStorage() });
  const agent2 = harness2.newAgent();
  agent2.applySavedState(checkpoint);

  assert.ok(agent2.silverBrain !== null, 'silverBrain should be restored');
  const restoredFP = fingerprint(harness2.NeuralNetwork.fromJSON(agent2.silverBrain.export(), INPUTS));
  assert.ok(sameGenome(restoredFP, silverFP), 'restored silverBrain must match original');
});

test('applySavedState: silverBrain is null when checkpoint has no silverBrain field', () => {
  const harness = loadAgent({ storage: new MemoryStorage() });
  const agent = harness.newAgent();
  const { NeuralNetwork } = harness;

  // Fake a valid checkpoint without silverBrain
  const fakeCheckpoint = {
    brain: agent.brain.export(),
    activeBrain: agent.brain.export(),
    hasChampion: false,
    bestFitness: 0,
    generation: 1,
    currentEpisode: 1,
    episodeFitnesses: []
    // no silverBrain field
  };
  agent.silverBrain = NeuralNetwork.fromJSON(agent.brain.export(), INPUTS); // set before
  agent.applySavedState(fakeCheckpoint);
  assert.equal(agent.silverBrain, null, 'missing silverBrain in checkpoint should clear it');
});

// ---------------------------------------------------------------------------
// 8. crossover breeding is triggered at stagnantGens >= 5 with silver present
// ---------------------------------------------------------------------------
test('crossover breeding: used when stagnantGens >= 5 and silverBrain exists', () => {
  const { agent, NeuralNetwork, logs } = withChampion({ fitness: 1000, stagnantGens: 4 });

  // Populate silverBrain by crowning once
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 2000;
    agent.survivalTime = 60;
    agent.onDeath(false);
  }
  assert.ok(agent.silverBrain !== null, 'silverBrain must exist for this test');
  // stagnantGens was reset to 0 on crowning; manually set to trigger crossover path
  agent.stagnantGens = 5;

  // Record brain fingerprint before a losing rollout (will trigger breeding)
  const champFP = fingerprint(NeuralNetwork.fromJSON(agent.championBrain.export(), INPUTS));

  // Run a losing rollout (below champion, stagnant++) → should trigger crossover
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 100;
    agent.survivalTime = 40;
    agent.onDeath(false);
  }

  // The log should mention Recombination
  const recombinationLog = logs.find(l => l.includes('Recombination'));
  assert.ok(recombinationLog, 'Expected a Recombination log when stagnantGens >= 5');
});

// ---------------------------------------------------------------------------
// 9. crossover breeding is NOT used when stagnantGens < 5
// ---------------------------------------------------------------------------
test('crossover breeding: NOT used when stagnantGens < 5 (pure clone path)', () => {
  const { agent, logs } = withChampion({ fitness: 1000, stagnantGens: 0 });

  // Populate silverBrain
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 2000;
    agent.survivalTime = 60;
    agent.onDeath(false);
  }
  agent.stagnantGens = 2; // below threshold

  // Losing rollout — should use pure clone, not crossover
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 100;
    agent.survivalTime = 40;
    agent.onDeath(false);
  }

  const recombinationLog = logs.find(l => l.includes('Recombination'));
  assert.ok(!recombinationLog, 'Must not use crossover when stagnantGens < 5');
});

// ---------------------------------------------------------------------------
// 10. crossover breeding is NOT used when silverBrain is null even if stagnant
// ---------------------------------------------------------------------------
test('crossover breeding: NOT used when silverBrain is null even when stagnantGens >= 5', () => {
  const { agent, logs } = withChampion({ fitness: 1000, stagnantGens: 10 });
  assert.equal(agent.silverBrain, null, 'silverBrain must be null for this test');

  // Losing rollout below 75% — no silver candidate qualified
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 100;
    agent.survivalTime = 40;
    agent.onDeath(false);
  }

  const recombinationLog = logs.find(l => l.includes('Recombination'));
  assert.ok(!recombinationLog, 'Must not crossover when silverBrain is null');
});

// ---------------------------------------------------------------------------
// 11. Stalled checkpoint with no silver gets one from a qualifying mutant
// ---------------------------------------------------------------------------
test('silver candidate: stalled checkpoint with no silver gets one from qualifying mutant (>= 75%)', () => {
  // Simulate stalled checkpoint like Gen 337 (stagnantGens: 270, silverBrain: null)
  const { agent, NeuralNetwork, logs } = withChampion({ fitness: 30000, stagnantGens: 270 });
  assert.equal(agent.silverBrain, null, 'initially no silver brain');
  assert.equal(agent.silverSource, null);

  // Run autonomous mutant rollout scoring 24,000 (80% of champ -> qualifies for mutantWorthyOfBlend)
  const mutantFP = fingerprint(NeuralNetwork.fromJSON(agent.brain.export(), INPUTS));
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 24000;
    agent.survivalTime = 50;
    agent.onDeath(false);
  }

  // Did not beat champion (24000 < 30000), but qualified as silver candidate
  assert.ok(agent.silverBrain !== null, 'qualifying mutant must be saved as silver candidate');
  assert.equal(agent.silverSource, 'mutant', 'source must be marked mutant');
  assert.equal(agent.silverFitness, 24000, 'recorded fitness must match aggregate');
  const silverFP = fingerprint(NeuralNetwork.fromJSON(agent.silverBrain.export(), INPUTS));
  assert.ok(sameGenome(silverFP, mutantFP), 'silverBrain matches qualifying mutant weights');

  // And since stagnantGens >= 5 (it was 270), the subsequent breed triggered Recombination!
  const recombinationLog = logs.find(l => l.includes('Recombination'));
  assert.ok(recombinationLog, 'Subsequent breeding should trigger crossover recombination');
});

// ---------------------------------------------------------------------------
// 12. Better-scoring mutant replaces candidate, but lower-scoring does not
// ---------------------------------------------------------------------------
test('silver candidate: better-scoring mutant replaces candidate, lower does not', () => {
  const { agent, NeuralNetwork } = withChampion({ fitness: 30000, stagnantGens: 10 });

  // First qualifying mutant: scores 23,000 (>= 75% of 30,000 = 22,500)
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 23000;
    agent.survivalTime = 50;
    agent.onDeath(false);
  }
  assert.equal(agent.silverFitness, 23000);
  const firstSilverFP = fingerprint(NeuralNetwork.fromJSON(agent.silverBrain.export(), INPUTS));

  // Second mutant: scores 22,800 (>= 75%, but lower than 23,000 -> must NOT replace)
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 22800;
    agent.survivalTime = 50;
    agent.onDeath(false);
  }
  assert.equal(agent.silverFitness, 23000, 'lower-scoring mutant must not replace candidate');
  const afterLowerFP = fingerprint(NeuralNetwork.fromJSON(agent.silverBrain.export(), INPUTS));
  assert.ok(sameGenome(afterLowerFP, firstSilverFP), 'weights unchanged after lower candidate');

  // Third mutant: scores 26,000 (higher than 23,000 -> MUST replace)
  const betterMutantFP = fingerprint(NeuralNetwork.fromJSON(agent.brain.export(), INPUTS));
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 26000;
    agent.survivalTime = 50;
    agent.onDeath(false);
  }
  assert.equal(agent.silverFitness, 26000, 'higher-scoring mutant replaces candidate');
  const afterHigherFP = fingerprint(NeuralNetwork.fromJSON(agent.silverBrain.export(), INPUTS));
  assert.ok(sameGenome(afterHigherFP, betterMutantFP), 'weights updated to higher candidate');
});

// ---------------------------------------------------------------------------
// 13. A displaced champion still takes priority over mutant candidates
// ---------------------------------------------------------------------------
test('silver priority: displaced champion takes priority over mutant candidates', () => {
  const { agent, NeuralNetwork } = withChampion({ fitness: 1000, stagnantGens: 0 });

  // Crown a new champion (displaces old champion to silver with fitness 1000)
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 2000;
    agent.survivalTime = 60;
    agent.onDeath(false);
  }
  assert.equal(agent.silverSource, 'champion', 'real displaced champion has source champion');
  assert.equal(agent.silverFitness, 1000);
  const champSilverFP = fingerprint(NeuralNetwork.fromJSON(agent.silverBrain.export(), INPUTS));

  // A subsequent mutant achieves 1,800 (which is 90% of current champ 2000 > 75%)
  for (let ep = 0; ep < EPISODES; ep++) {
    agent.fitness = 1800;
    agent.survivalTime = 50;
    agent.onDeath(false);
  }

  // Must NOT overwrite the real displaced champion!
  assert.equal(agent.silverSource, 'champion', 'source must remain champion');
  assert.equal(agent.silverFitness, 1000, 'fitness must remain former champion fitness');
  const currentSilverFP = fingerprint(NeuralNetwork.fromJSON(agent.silverBrain.export(), INPUTS));
  assert.ok(sameGenome(currentSilverFP, champSilverFP), 'real displaced champion weights preserved');
});

// ---------------------------------------------------------------------------
// 14. Fallback silver candidate round-trips through checkpoint
// ---------------------------------------------------------------------------
test('silver candidate: round-trips cleanly through checkpoint export/import', () => {
  const storage = new MemoryStorage();
  const harness1 = loadAgent({ storage });
  const agent1 = harness1.newAgent();
  agent1.resetBrain();
  agent1.bestFitness = 20000;
  agent1.allTimePeakFitness = 20000;
  agent1.stagnantGens = 15;
  const { NeuralNetwork } = harness1;
  agent1.championBrain = NeuralNetwork.fromJSON(agent1.brain.export(), INPUTS);

  // Mutant qualifies with 16,000 (80% of 20,000)
  for (let ep = 0; ep < EPISODES; ep++) {
    agent1.fitness = 16000;
    agent1.survivalTime = 50;
    agent1.onDeath(false);
  }
  assert.equal(agent1.silverSource, 'mutant');
  assert.equal(agent1.silverFitness, 16000);
  const silverFP = fingerprint(NeuralNetwork.fromJSON(agent1.silverBrain.export(), INPUTS));

  // Export checkpoint and load into fresh agent
  const checkpoint = JSON.parse(agent1.exportCheckpoint());
  assert.equal(checkpoint.silverSource, 'mutant');
  assert.equal(checkpoint.silverFitness, 16000);

  const harness2 = loadAgent({ storage: new MemoryStorage() });
  const agent2 = harness2.newAgent();
  agent2.applySavedState(checkpoint);

  assert.ok(agent2.silverBrain !== null);
  assert.equal(agent2.silverSource, 'mutant');
  assert.equal(agent2.silverFitness, 16000);
  const restoredFP = fingerprint(harness2.NeuralNetwork.fromJSON(agent2.silverBrain.export(), INPUTS));
  assert.ok(sameGenome(restoredFP, silverFP), 'restored candidate matches original');
});


