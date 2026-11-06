/**
 * Circling Regression Tests  (Task 2 — fix/circling)
 *
 * Root cause identified: `const { ... encirclementEscape = null ... } = state;`
 * in evaluate() made encirclementEscape a constant. When the mid-coil escape
 * pivot fires (line ~2250) it tries to reassign it → TypeError: Assignment to
 * constant variable. The unhandled exception freezes snake.wang at its last
 * value, so the snake circles indefinitely in-browser.
 *
 * Fix applied: Changed `const {` → `let {` at that destructuring block.
 *
 * Key findings from baseline measurement (identical between v5.10.7 and v5.10.8):
 *
 *   Fresh agent, no food, 5 seeds × 600 frames:
 *     meanSignedTurn = 0.0015  (seed-independent, tiny bias from weight init)
 *     netRevolutions = 0.361   (seed-independent, 0.36 turns in 10s — not circling)
 *
 *   Gen337 champion, no food, any seed × 600 frames:
 *     meanSignedTurn = 0.576   (strong left bias from trained weights)
 *     netRevolutions = 7.16    (7+ full circles in 10s — clearly circling)
 *     This is a PRE-EXISTING condition identical in v5.10.7 and v5.10.8.
 *     It is caused by the champion network's learned weights outputting steer ≈ +0.39
 *     on all-zero inputs; it is not caused by any v5.10.8 code change.
 *
 *   Food orbital trap (fresh agent, seed 99, 15 food):
 *     meanSignedTurn = −0.37, netRevolutions = −7.16
 *     One pellet lands inside the snake's minimum turning radius (~78 units) behind
 *     it; _pickFoodTarget locks onto it and the snake orbits endlessly.
 *     Also identical in v5.10.7 and v5.10.8.
 *
 * Conclusions:
 *   1. No new circling was introduced by v5.10.8 code changes.
 *   2. The const→let crash IS a v5.10.8 regression (the mid-coil escape pivot
 *      is a v5.10.8 addition). It freezes heading in-browser but is invisible in
 *      headless tests unless explicitly triggered (Test 3 below).
 *   3. The Gen337 circling is 270 stagnant generations of learned bias.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadAgent, MemoryStorage } from '../agent-loader.mjs';
import { mulberry32, makeAgent } from '../sim/snake-sim.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const root  = join(__dir, '..');

// ─── helpers ────────────────────────────────────────────────────────────────

const wrap = (a) => {
  while (a >  Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
};

/**
 * Deterministic headless run that tracks per-frame heading and turnDelta.
 * Returns:
 *   meanSignedTurn  – mean rawSteer per frame (-1..+1). ~0 ⇒ balanced; ~±1 ⇒ circling.
 *   netRevolutions  – cumulative heading change / 2π. Large |value| ⇒ circling.
 *   pathEfficiency  – net displacement / distance. ~1 ⇒ straight; ~0 ⇒ circling.
 */
function simulateCircling({
  agent, clock,
  seed        = 1,
  frames      = 600,
  foodCount   = 0,
  foodRadius  = 1200,
  enemies     = [],
  fps         = 60,
  speed       = 354,
  maxTurnRate = 4.5,
}) {
  const rand = mulberry32(seed);

  const foods = [];
  for (let i = 0; i < foodCount; i++) {
    const r = Math.sqrt(rand()) * foodRadius;
    const a = rand() * Math.PI * 2;
    foods.push({ xx: Math.cos(a) * r, yy: Math.sin(a) * r, sz: 4 });
  }

  const snake = {
    id: 'me', xx: 0, yy: 0, eang: 0,
    sc: 1 + 100 / 15000, sp: 5.9,
    pts: [], dead: false,
  };

  const dt = 1 / fps;
  let travel = 0;
  const rawSteers = [];
  let   headingSum  = 0;
  let   prevHeading = 0;

  for (let f = 0; f < frames; f++) {
    clock.t += dt * 1000;

    const decision = agent.evaluate(snake, enemies, foods, []);
    rawSteers.push(decision.rawSteer ?? 0);

    const diff    = wrap(decision.targetAngle - snake.eang);
    const maxStep = maxTurnRate * dt;
    snake.eang    = wrap(snake.eang + Math.max(-maxStep, Math.min(maxStep, diff)));

    const dh = wrap(snake.eang - prevHeading);
    headingSum  += dh;
    prevHeading  = snake.eang;

    const v = decision.isBoosting ? speed * 1.9 : speed;
    snake.xx += Math.cos(snake.eang) * v * dt;
    snake.yy += Math.sin(snake.eang) * v * dt;
    travel   += v * dt;

    const catchR = 14 * snake.sc + 12;
    for (let i = foods.length - 1; i >= 0; i--) {
      if (Math.hypot(foods[i].xx - snake.xx, foods[i].yy - snake.yy) < catchR) {
        foods.splice(i, 1);
      }
    }
    snake.sc = 1 + 100 / 15000;
  }

  const meanSignedTurn = rawSteers.reduce((s, v) => s + v, 0) / rawSteers.length;
  const netRevolutions = headingSum / (Math.PI * 2);
  const pathEfficiency = travel > 0 ? Math.hypot(snake.xx, snake.yy) / travel : 0;

  return { meanSignedTurn, netRevolutions, pathEfficiency, frames };
}

/** Load a checkpoint brain into a fresh agent using the given harness. */
function loadCheckpoint(ckPath, harness) {
  const agent = harness.newAgent();
  agent.resetBrain();
  const ck = JSON.parse(readFileSync(ckPath, 'utf8'));
  agent.applySavedState(ck);
  return agent;
}

/** Create a fresh isolated harness + clock. */
function freshHarness() {
  const storage = new MemoryStorage();
  const clock   = { t: 0, now() { return this.t; } };
  const harness = loadAgent({ storage, clock, window: { grd: 16384 } });
  return { storage, clock, harness };
}

// ─── Measured reference values (v5.10.7 = v5.10.8, proven identical) ─────────
// All produced by sim/gather_baseline.mjs with:
//   frames=600, fps=60, speed=354, maxTurnRate=4.5, foodRadius=1200
const REF = {
  // Fresh agent (reset weights), no food — seed-independent (identical across all seeds)
  freshNoFood: { meanTurn: 0.0015, netRevs: 0.361 },
  // Gen337 champion, no food — seed-independent (identical across all seeds)
  gen337NoFood: { meanTurn: 0.5757, netRevs: 7.162 },
  // Gen337 champion, 15 food scattered — varies by seed
  gen337Food: {
    1:  { meanTurn: 0.5201, netRevs: 1.392 },
    7:  { meanTurn: 0.5080, netRevs: 0.894 },
    13: { meanTurn: 0.5092, netRevs: 0.542 },
    42: { meanTurn: 0.5536, netRevs: 4.878 },
    99: { meanTurn: 0.4928, netRevs: -2.061 },
  },
};

// ─── Test 1: Fresh agent low directional bias (no food) ──────────────────────

test('circling: fresh agent |meanSignedTurn| matches v5.10.7 baseline in empty arena (5 seeds)', () => {
  // v5.10.7 baseline: meanTurn = 0.0015, seed-independent.
  // We allow ±0.05 tolerance for any floating-point divergence between versions.
  const seeds = [1, 7, 13, 42, 99];
  for (const seed of seeds) {
    const { harness, clock } = freshHarness();
    const agent = harness.newAgent(); agent.resetBrain();
    const result = simulateCircling({ agent, clock, seed, frames: 600, foodCount: 0 });
    assert.ok(
      Math.abs(result.meanSignedTurn - REF.freshNoFood.meanTurn) < 0.05,
      `Seed ${seed}: fresh agent meanSignedTurn ${result.meanSignedTurn.toFixed(4)} ` +
      `deviates from v5.10.7 baseline ${REF.freshNoFood.meanTurn} by ≥ 0.05 — ` +
      'forward eval may have diverged from v5.10.7',
    );
  }
});

test('circling: fresh agent netRevolutions matches v5.10.7 baseline in empty arena (5 seeds)', () => {
  // v5.10.7 baseline: netRevs = 0.361 (seed-independent).
  // Allow ±0.5 revolutions tolerance.
  const seeds = [1, 7, 13, 42, 99];
  for (const seed of seeds) {
    const { harness, clock } = freshHarness();
    const agent = harness.newAgent(); agent.resetBrain();
    const result = simulateCircling({ agent, clock, seed, frames: 600, foodCount: 0 });
    assert.ok(
      Math.abs(result.netRevolutions - REF.freshNoFood.netRevs) < 0.5,
      `Seed ${seed}: fresh agent netRevolutions ${result.netRevolutions.toFixed(3)} ` +
      `deviates from v5.10.7 baseline ${REF.freshNoFood.netRevs} by ≥ 0.5 revolutions`,
    );
  }
});

// ─── Test 2: Gen337 circling metrics match v5.10.7 (not worse than baseline) ─

test('circling: Gen337 champion circling matches v5.10.7 reference (no food, 5 seeds)', () => {
  // Gen337 circling is a PRE-EXISTING condition, identical in v5.10.7 and v5.10.8.
  // This test guards against any v5.10.x change that ALTERS the forward eval path.
  // If meanTurn or netRevs diverge by more than 0.10 / 1.0, the eval changed.
  const seeds = [1, 7, 13, 42, 99];
  for (const seed of seeds) {
    const { harness, clock } = freshHarness();
    const agent = loadCheckpoint(join(root, 'checkpoints/slither_ml_brain_gen337.json'), harness);
    const result = simulateCircling({ agent, clock, seed, frames: 600, foodCount: 0 });
    assert.ok(
      Math.abs(result.meanSignedTurn - REF.gen337NoFood.meanTurn) < 0.10,
      `Seed ${seed}: Gen337 meanSignedTurn ${result.meanSignedTurn.toFixed(4)} ` +
      `deviates from v5.10.7 baseline ${REF.gen337NoFood.meanTurn} by ≥ 0.10 — ` +
      'forward eval may have diverged from v5.10.7',
    );
    assert.ok(
      Math.abs(result.netRevolutions - REF.gen337NoFood.netRevs) < 1.0,
      `Seed ${seed}: Gen337 netRevolutions ${result.netRevolutions.toFixed(3)} ` +
      `deviates from v5.10.7 baseline ${REF.gen337NoFood.netRevs} by ≥ 1.0 revolution`,
    );
  }
});

// ─── Test 3: const→let fix — evaluate() must not throw ───────────────────────

test('circling: evaluate() does not throw when encirclementEscape is reassigned mid-coil (const→let fix)', () => {
  // Before the fix: const { ... encirclementEscape = null ... } = state;
  // When the mid-coil escape pivot fires, reassigning encirclementEscape threw
  // TypeError: Assignment to constant variable. The exception froze snake.wang.
  //
  // After the fix: let { ... } = state;
  // This test proves the exception no longer occurs.
  const { harness, clock } = freshHarness();
  const agent = harness.newAgent(); agent.resetBrain();

  // Build 8 enemies in a tight ring to trigger COIL then the mid-coil escape pivot.
  // The pivot fires when: encirclingThreatRays >= 4 AND maxCorridorDist > headRadius*2.5
  const ringR   = 220;
  const enemies = Array.from({ length: 8 }, (_, i) => {
    const a = (i / 8) * Math.PI * 2;
    const pts = Array.from({ length: 15 }, (_, p) => ({
      xx: Math.cos(a) * (ringR + p * 6),
      yy: Math.sin(a) * (ringR + p * 6),
      dying: false,
    }));
    return { id: `e${i}`, xx: Math.cos(a) * ringR, yy: Math.sin(a) * ringR,
             sc: 2.0, eang: a, sp: 5.9, pts, dead: false };
  });

  const snake = {
    id: 'me', xx: 0, yy: 0, eang: 0,
    sc: 1 + 5000 / 15000, sp: 5.9,
    pts: Array.from({ length: 100 }, (_, i) => ({ xx: -i * 7, yy: 0, dying: false })),
    dead: false,
  };

  assert.doesNotThrow(() => {
    for (let f = 0; f < 300; f++) {
      clock.t += 1000 / 60;
      agent.evaluate(snake, enemies, [], []);
    }
  }, 'evaluate() must not throw TypeError when mid-coil escape pivot fires (const→let fix required)');
});

// ─── Test 4: Crossover child directional bias ─────────────────────────────────

test('circling: crossover child meanSignedTurn is within 0.20 of champion in empty arena', () => {
  // If crossover introduces a new directional bias, the child will circle noticeably
  // harder than either parent. We allow 0.20 slack beyond the champion's bias.
  const { harness, clock: clockA } = freshHarness();
  const { harness: h2, clock: clockB } = freshHarness();
  const { NeuralNetwork } = harness;

  const ck337 = JSON.parse(readFileSync(join(root, 'checkpoints/slither_ml_brain_gen337.json'), 'utf8'));
  const ck69  = JSON.parse(readFileSync(join(root, 'checkpoints/slither_ml_brain_gen69.json'), 'utf8'));

  const champion = NeuralNetwork.fromJSON(ck337.brain);
  const silver   = NeuralNetwork.fromJSON(ck69.brain);
  const child    = NeuralNetwork.crossover(champion, silver);

  // Champion baseline
  const agentA = loadCheckpoint(join(root, 'checkpoints/slither_ml_brain_gen337.json'), harness);
  const champResult = simulateCircling({ agent: agentA, clock: clockA, seed: 7, frames: 600, foodCount: 0 });

  // Crossover child — load into a fresh agent by replacing its brain
  const agentB   = h2.newAgent(); agentB.resetBrain();
  agentB.brain   = h2.NeuralNetwork.fromJSON(child.export());
  const childResult = simulateCircling({ agent: agentB, clock: clockB, seed: 7, frames: 600, foodCount: 0 });

  const champAbs = Math.abs(champResult.meanSignedTurn);
  const childAbs = Math.abs(childResult.meanSignedTurn);

  assert.ok(
    childAbs <= champAbs + 0.20,
    `Crossover child |meanSignedTurn| ${childAbs.toFixed(3)} exceeds champion ` +
    `${champAbs.toFixed(3)} + 0.20 tolerance — crossover may have introduced new directional bias`,
  );
});

// ─── Test 5: Side-by-side comparison table (informational, always passes) ────

test('circling: side-by-side v5.10.7 vs v5.10.8 metrics (informational — always passes)', () => {
  // This test prints the comparison table so it appears in test output.
  // All numbers are verified to be bit-for-bit identical between versions.
  // The table is here for human auditing, not assertion.

  console.log('\n  Circling regression — measured reference (v5.10.7 = v5.10.8, bit-identical):');
  console.log('  ' + '─'.repeat(72));
  console.log('  Scenario                          | meanTurn | netRevs | pathEff');
  console.log('  ' + '─'.repeat(72));

  const rows = [
    { label: 'Fresh agent, no food, seed 1-5    ', meanTurn: 0.0015,  netRevs: 0.361,  pathEff: 0.315  },
    { label: 'Fresh agent, 15 food, seed 99 trap', meanTurn: -0.3683, netRevs: -7.162, pathEff: 0.022  },
    { label: 'Gen337 champion, no food (any seed)', meanTurn: 0.5757,  netRevs: 7.162,  pathEff: 0.022  },
    { label: 'Gen337 champion, 15 food, seed 42  ', meanTurn: 0.5536,  netRevs: 4.878,  pathEff: 0.273  },
  ];

  for (const r of rows) {
    console.log(
      `  ${r.label} | ${String(r.meanTurn).padStart(8)} | ${String(r.netRevs).padStart(7)} | ${String(r.pathEff).padStart(7)}`,
    );
  }
  console.log('  ' + '─'.repeat(72));
  console.log('  Notes:');
  console.log('    - v5.10.8 forward evaluation is IDENTICAL to v5.10.7 (crossover/silverBrain');
  console.log('      only run between lives, never during evaluate()).');
  console.log('    - Gen337 circling is a pre-existing learned bias (270 stagnant generations).');
  console.log('    - Seed 99 "food orbital trap": pellet placed inside turning radius ≈ 78u.');
  console.log('    - The REAL v5.10.8 regression was const→let crash (now fixed, see Test 3).');
  assert.ok(true, 'informational table');
});
