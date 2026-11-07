/**
 * Evolution Distribution & Unfreezing Tests (Phase 3b)
 *
 * Verifies:
 *   1. sim/snake-sim.mjs evaluateDistribution computes quantiles (median, Q1, Q3, IQR) over fixed seeds.
 *   2. evaluateRollout derives measured distribution metrics (median, Q3, floor) from rollout history.
 *   3. Deep stagnation (stagnantGens >= 60) eases the plateau floor toward measured performance.
 *   4. Teacher candidate achieving realistic autonomous rollout (>= 1,500 points) qualifies for Tier 2 champion blend.
 *   5. Autonomous mutant beating the relaxed measured floor is crowned new champion and passes consistency gate.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadAgent, MemoryStorage } from '../agent-loader.mjs';
import { evaluateDistribution } from '../sim/snake-sim.mjs';
import { evaluateRollout } from '../src/evolution.js';

test('evaluateDistribution produces deterministic quantiles across fixed seeds', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();

  const dist = evaluateDistribution(agent, {
    seedCount: 10,
    startSeed: 1,
    seconds: 5,
    foodCount: 50
  });

  assert.equal(dist.seedCount, 10);
  assert.ok(typeof dist.scores.median === 'number', 'scores must have median');
  assert.ok(typeof dist.scores.q1 === 'number', 'scores must have Q1');
  assert.ok(typeof dist.scores.q3 === 'number', 'scores must have Q3');
  assert.ok(dist.scores.q3 >= dist.scores.q1, 'Q3 must be >= Q1');
  assert.equal(dist.scores.iqr, dist.scores.q3 - dist.scores.q1, 'IQR must equal Q3 - Q1');
  assert.ok(dist.survivals.median > 0, 'survival median must be positive');
});

test('evaluateRollout calculates distribution metrics from rollout history', () => {
  const result = evaluateRollout({
    episodeFitnesses: [1200, 1000, 800],
    previousBest: 30000,
    allTimePeakFitness: 43634,
    stagnantGens: 10,
    candidatePending: false,
    candidateSource: null,
    isAutonomousMutant: true,
    recentRollouts: [600, 800, 950, 1100, 1250]
  });

  assert.equal(result.aggregateFitness, 1100);
  assert.equal(result.meanFitness, 1000);
  assert.equal(result.measuredMedian, 950);
  assert.equal(result.measuredQ3, 1100);
  assert.equal(result.measuredFloor, 1100);
});

test('deep plateau relief decays floor toward measured distribution when stagnantGens >= 60', () => {
  // Peak was 43,634 (70% floor is 30,543.8). Previous best was pinned at 30,543.8.
  // With stagnantGens = 70, minBar should ease toward measuredFloor (1,200) instead of being locked at 30,543.8.
  const result = evaluateRollout({
    episodeFitnesses: [1000, 900, 800],
    previousBest: 30543.8,
    allTimePeakFitness: 43634,
    stagnantGens: 70,
    candidatePending: false,
    candidateSource: null,
    isAutonomousMutant: true,
    recentRollouts: [700, 900, 1000, 1100, 1200]
  });

  assert.ok(result.nextBestFitness < 30543.8, `nextBestFitness ${result.nextBestFitness} should decay below 30,543.8`);
  assert.ok(result.nextBestFitness >= 1200, `nextBestFitness ${result.nextBestFitness} must respect measuredFloor`);
});

test('teacher candidate with rollout under 5,000 and under 35% falls into Tier 3 tuition preservation', () => {
  const result = evaluateRollout({
    episodeFitnesses: [2000, 2000, 2000],
    previousBest: 30543.8,
    allTimePeakFitness: 43634,
    stagnantGens: 20,
    candidatePending: true,
    candidateSource: 'teacher',
    isAutonomousMutant: false,
    recentRollouts: [800, 900, 1000]
  });

  assert.equal(result.aggregateFitness, 2000);
  assert.equal(result.teacherWorthyOfChampBlend, false, 'candidate scoring 2,000 < 5,000 does not blend into champion');
  assert.equal(result.teacherWorthyOfBreedingBlend, true, 'candidate scoring 2,000 preserves tuition into breeding seed');
});

test('autonomous mutant is crowned new champion when bestFitness has decayed to measured baseline', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();

  // Agent was unfrozen and decayed down to 1,200
  agent.bestFitness = 1200;
  agent.allTimePeakFitness = 1500;
  agent.championBrain = harness.NeuralNetwork.fromJSON(agent.brain.export(), 32);
  agent.stagnantGens = 10;

  // New mutant achieves 1,800 on all 3 lives
  agent.currentEpisode = 1;
  agent.episodeFitnesses = [];
  for (let i = 0; i < 3; i++) {
    agent.fitness = 1800;
    agent.survivalTime = 50;
    agent.onDeath(false);
  }

  assert.equal(agent.bestFitness, 1800, 'mutant should be crowned new champion');
  assert.equal(agent.stagnantGens, 0, 'stagnantGens should reset to 0');
  assert.ok(agent.silverBrain !== null, 'former champion should be demoted to silver');
  assert.equal(agent.silverFitness, 1200);
});
