/**
 * Behaviour tests for the two reported gameplay faults:
 *   1. the snake boosted almost continuously and bled its length away
 *   2. it ignored food instead of growing
 *
 * These run the real agent through the headless simulator, so they assert on
 * outcomes (pellets eaten, share of frames boosting) rather than on code shape.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { makeAgent, simulate } from '../sim/snake-sim.mjs';
import { loadAgent, MemoryStorage } from '../agent-loader.mjs';

const GRD = 16384;

function inputsFor(snakePos, foods) {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: snakePos.x, yy: snakePos.y, eang: 0, sc: 1.02, sp: 5.9, pts: [] };
  return { agent, state: agent._extractInputs(snake, [], foods, [], 0.0167, clock.t) };
}

const scattered = (n, radius = 2600, seed = 42) => {
  let s = seed;
  const rand = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  return Array.from({ length: n }, () => {
    const r = Math.sqrt(rand()) * radius;
    const a = rand() * Math.PI * 2;
    return { xx: Math.cos(a) * r, yy: Math.sin(a) * r, sz: 4 };
  });
};

// ---------------------------------------------------------------------------
// 1. boost: must not bleed length
// ---------------------------------------------------------------------------
test('it does not boost while foraging ordinary pellets', () => {
  const r = simulate({ foodLayout: 'uniform', foodCount: 800, seed: 7, seconds: 60 });
  assert.ok(r.boostDuty < 0.02, `boost duty ${(r.boostDuty * 100).toFixed(1)}% should be ~0 for plain pellets`);
  assert.ok(r.scoreDelta >= 0, `score should not shrink: ${r.scoreDelta}`);
});

test('the boost governor refuses a sprint when it is unsafe', () => {
  const run = (over) => {
    const { agent: a, clock } = makeAgent(1); // must use the agent's own virtual clock
    const inputs = new Float32Array(32);
    if (over.frontBlocked) { inputs[7] = 0.9; inputs[8] = 0.9; inputs[9] = 0.9; }
    if (over.congested) inputs[31] = 0.9;
    a._extractInputs = () => ({
      inputs,
      minObstacleDist: 600,
      frontObstacleDist: over.frontBlocked ? 20 : 600,
      leftClearance: 100,
      rightClearance: 100,
      forwardObstacleDanger: false,
      normWallDist: 0.1,
      diffCenterAngle: 0,
      headRadius: 20,
      currentAng: 0,
      currentScore: over.score ?? 200,
    });
    a.peakLength = over.peak ?? 0;
    a.detectedFoods = over.target === false ? [] : [{ dx: 200, dy: 0, dist: 200, sz: 20, isPrey: false }];

    let boosted = 0;
    const frames = 20 * 60;
    for (let i = 0; i < frames; i++) {
      clock.t += 1000 / 60;
      const snake = { xx: 0, yy: 0, sc: 1 + (over.score ?? 200) / 15000, eang: 0, sp: 5.9 };
      if (a.evaluate(snake, [], [], []).isBoosting) boosted++;
    }
    return boosted / frames;
  };

  const withTarget = run({});
  assert.ok(withTarget > 0.05, `a valuable drop ahead should be worth a sprint, got ${withTarget}`);
  assert.ok(withTarget < 0.45, `even then the duty cycle is capped by burst/rest, got ${withTarget}`);
  assert.equal(run({ frontBlocked: true }), 0, 'never sprint into an obstacle');
  assert.equal(run({ score: 50 }), 0, 'never sprint below the length floor');
  assert.equal(run({ score: 200, peak: 300 }), 0, 'never sprint while depleted from peak');
  assert.equal(run({ congested: true }), 0, 'never sprint in heavy traffic');
  assert.equal(run({ target: false }), 0, 'ordinary pellets are not worth a sprint');
});

// ---------------------------------------------------------------------------
// 2. food seeking
// ---------------------------------------------------------------------------
test('food radar is scaled so it is actually visible to the network', () => {
  const dense = inputsFor({ x: 0, y: 0 }, scattered(2000));
  const sector = Array.from(dense.state.inputs.slice(16, 24));
  assert.ok(Math.max(...sector) > 0.3, `dense food should register, got ${Math.max(...sector).toFixed(3)}`);

  const sparse = inputsFor({ x: 0, y: 0 }, scattered(30, 1200));
  assert.ok(Math.max(...Array.from(sparse.state.inputs.slice(16, 24))) < 0.35, 'a few pellets must not saturate the radar');
});

test('it actually eats: intake is high in a food field', () => {
  const r = simulate({ foodLayout: 'uniform', foodCount: 800, seed: 7, seconds: 60 });
  assert.ok(r.pelletsPerMinute > 40, `ate ${r.pelletsPerMinute.toFixed(1)} pellets/min`);
  assert.ok(r.scoreDelta > 40, `grew by ${r.scoreDelta.toFixed(0)} score`);
});

test('it steers toward a clump of food placed to the side', () => {
  const r = simulate({ foodLayout: 'left', foodCount: 400, seed: 13, seconds: 30 });
  assert.ok(r.pelletsPerMinute > 40, `ate ${r.pelletsPerMinute.toFixed(1)} pellets/min`);
  assert.ok(r.pointedAtFood > 0.5, `only ${(r.pointedAtFood * 100).toFixed(0)}% of frames aimed at food`);
});

test('dead snake remains beyond obstacle sensor range are detected and prioritized', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.02, sp: 5.9, pts: [] };
  // Pre-fill with 100 small ambient pellets
  const foods = Array.from({ length: 100 }, (_, i) => ({
    xx: 200 + (i % 10) * 15,
    yy: -150 + Math.floor(i / 10) * 30,
    sz: 3
  }));
  // Dead snake carcass exploded 850 units ahead (beyond the ~520 obstacle sensorRange)
  foods.push({ xx: 850, yy: 0, sz: 22 });

  agent._extractInputs(snake, [], foods, [], 0.0167, clock.t);

  // Carcass drop must be detected in detectedFoods
  const corpse = agent.detectedFoods.find(f => f.sz === 22);
  assert.ok(corpse, 'corpse pellet at dist 850 must be perceived in detectedFoods');
  assert.equal(corpse.dist, 850);

  // Foraging assist should target the high-value carcass
  const target = agent._pickFoodTarget(0);
  assert.ok(target, 'should pick a food target');
  assert.equal(target.value, 8, 'should prioritize high-value dead snake drop');
  assert.equal(target.dist, 850);
});

// ---------------------------------------------------------------------------
// 3. the arena-centre trap that made it circle
// ---------------------------------------------------------------------------
test('the arena-centre channel is neutral at the origin', () => {
  const { state } = inputsFor({ x: 0, y: 0 }, []);
  // atan2(-0,-0) is +/-pi; ungated this commanded a permanent U-turn
  assert.equal(state.inputs[25], 0, 'centre bearing must be neutralised at the origin');
});

test('the arena-centre channel is active near the wall', () => {
  const { state } = inputsFor({ x: GRD * 0.9, y: 0 }, []);
  assert.ok(Math.abs(state.inputs[25]) > 0.2, `expected a strong centre signal near the wall, got ${state.inputs[25]}`);
});

test('with nothing to chase it flies straight instead of spiralling', () => {
  const r = simulate({ foodLayout: 'none', foodCount: 0, seed: 17, seconds: 30 });
  assert.ok(r.pathEfficiency > 0.9, `path efficiency ${r.pathEfficiency.toFixed(2)} should be near 1.0`);
});

// ---------------------------------------------------------------------------
// 4. forward-only collision urgency
// ---------------------------------------------------------------------------
test('an obstacle behind the snake does not count as forward danger', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.02, sp: 5.9, pts: [] };
  // build a body trailing straight behind the head
  for (let i = 0; i < 80; i++) snake.pts.push({ xx: -12 * (i + 1), yy: 0, dying: false });
  const state = agent._extractInputs(snake, [], [], [], 0.0167, clock.t);

  assert.ok(state.minObstacleDist < 200, 'the tail should register on the 360-degree minimum');
  assert.ok(state.frontObstacleDist > state.minObstacleDist * 2, 'but the forward cone should be clear');
});

// ---------------------------------------------------------------------------
// 5. tactical combat stances: defend and attack
// ---------------------------------------------------------------------------
test('anti-cut-off defence: banks away from an intercepting enemy and stops boosting', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.02, sp: 5.9, pts: [] };
  // Enemy on our left flank (xx: 120, yy: 60) moving across our nose toward the right
  const enemies = [{
    id: 'e1',
    xx: 120,
    yy: 60,
    eang: -Math.PI / 3, // moving downward-right across our path
    sp: 9.0,
    sc: 1.1,
    dead: false,
    pts: []
  }];

  const decision = agent.evaluate(snake, enemies, [], []);
  assert.equal(decision.stance, 'DEFEND', 'should trigger DEFEND stance on cut-off attempt');
  assert.equal(decision.isBoosting, false, 'must not boost into a cut-off');
  assert.ok(decision.targetAngle < 0, 'must bank away to the right (targetAngle < 0) when cut from left');
});

test('predator attack strike: intercepts a vulnerable enemy with sprint boost', () => {
  const { agent, clock } = makeAgent(1);
  // High score, full boost stamina
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.25, sp: 5.9, sct: 250, pts: [] };
  agent.previousScore = 250;
  agent.peakLength = 250;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // A vulnerable enemy traveling parallel in front-right
  const enemies = [{
    id: 'e_prey',
    xx: 180,
    yy: 70,
    eang: 0.1,
    sp: 4.5,
    sc: 0.9,
    dead: false,
    pts: []
  }];

  const decision = agent.evaluate(snake, enemies, [], []);
  assert.equal(decision.stance, 'ATTACK', 'should trigger ATTACK stance on vulnerable target');
  assert.equal(decision.isBoosting, true, 'must sprint to cut off target');
  assert.ok(decision.targetAngle > 0, 'should lead target toward intercept point');
});

test('dense body sampling: an unbroken wall of 300 segments registers collision without ray tunneling', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, pts: [] };
  // Enemy with 300 segments placed as a vertical wall 180 units ahead
  const enemyPts = [];
  for (let i = -150; i <= 150; i++) {
    enemyPts.push({ xx: 180, yy: i * 8, dying: false });
  }
  const enemies = [{
    id: 'e_wall',
    xx: 180,
    yy: 1200,
    eang: 0,
    sp: 5.9,
    sc: 1.0,
    dead: false,
    pts: enemyPts
  }];

  const state = agent._extractInputs(snake, enemies, [], [], 0.0167, clock.t);
  assert.ok(state.inputs[8] > 0.5, `ray 8 must detect the body wall, got ${state.inputs[8]}`);
  assert.ok(state.frontObstacleDist < 200, `frontObstacleDist should be ~180, got ${state.frontObstacleDist}`);
});

// ---------------------------------------------------------------------------
// 6. multi-repo synthesized features (coil defense, boundary guard, kills, anti-stagnation)
// ---------------------------------------------------------------------------
test('coil defense: enters COIL stance and loops in protective circle when pinched by multiple threats', () => {
  const { agent } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.3, sp: 5.9, sct: 450, pts: [] };
  agent.previousScore = 450;
  agent.peakLength = 450;

  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    inputs[29] = 0.85; // high left threat
    inputs[30] = 0.85; // high right threat
    inputs[31] = 0.50; // multiple nearby enemies
    return {
      inputs,
      minObstacleDist: 35,
      frontObstacleDist: 40,
      leftClearance: 40,
      rightClearance: 40,
      forwardObstacleDanger: true,
      normWallDist: 0.1,
      diffCenterAngle: 0,
      headRadius: 18,
      currentAng: 0,
      currentScore: 450,
      cutOffThreat: null,
      attackCandidate: null
    };
  };

  const decision = agent.evaluate(snake, [], [], []);
  assert.equal(decision.stance, 'COIL', 'should trigger COIL stance when pinched by threats');
  assert.equal(decision.isBoosting, false, 'must not boost while coiling');
  assert.ok(decision.targetAngle > 0, 'commands continuous tight circular turn');
});

test('arena boundary defense: banks hard toward center and disables boost when near death wall', () => {
  const { agent } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, pts: [] };

  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    inputs[24] = 0.95; // very close to wall
    return {
      inputs,
      minObstacleDist: 400,
      frontObstacleDist: 400,
      leftClearance: 200,
      rightClearance: 200,
      forwardObstacleDanger: false,
      normWallDist: 0.95,
      diffCenterAngle: 1.2,
      headRadius: 14,
      currentAng: 0,
      currentScore: 100,
      cutOffThreat: null,
      attackCandidate: null
    };
  };

  const decision = agent.evaluate(snake, [], [], []);
  assert.equal(decision.stance, 'DEFEND', 'should trigger DEFEND stance near death wall');
  assert.equal(decision.isBoosting, false, 'must not boost into the death wall');
  assert.ok(decision.targetAngle > 0, 'must steer toward arena center');
});

test('confirmed kill detection: tracks vanished nearby enemy and awards kill bounty', () => {
  const { agent } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, pts: [] };

  const enemy1 = { id: 'e1', xx: 100, yy: 50, eang: 0, sp: 5.9, sc: 1.0, dead: false, pts: [] };
  agent._extractInputs(snake, [enemy1], [], [], 0.0167, 1000);
  assert.equal(agent.kills, 0, 'no kills yet');

  const initialFit = agent.fitness;
  agent._extractInputs(snake, [], [], [], 0.0167, 1050);
  assert.equal(agent.kills, 1, 'must detect confirmed kill #1');
  assert.ok(agent.fitness >= initialFit + 150.0, 'must award +150 fitness bounty');
});

test('anti-stagnation adaptive mutation: escalates mutation rate after 3 stagnant generations', () => {
  const { agent } = makeAgent(1);
  agent.bestFitness = 5000;
  agent.championBrain = agent.brain;

  for (let g = 0; g < 3; g++) {
    agent._recordEvaluationEpisode(100);
    agent._recordEvaluationEpisode(100);
    agent._recordEvaluationEpisode(100);
  }

  assert.ok(agent.stagnantGens >= 3, `stagnantGens should be >= 3, got ${agent.stagnantGens}`);
});

test('high-fitness annealed mutation: scales down mutation rates when bestFitness > 30000', () => {
  const { agent, harness } = makeAgent(1);
  agent.bestFitness = 35000;
  agent.championBrain = agent.brain;
  agent.stagnantGens = 0;

  let capturedRate = 0;
  let capturedStrength = 0;
  const originalMutate = harness.NeuralNetwork.prototype.mutate;
  harness.NeuralNetwork.prototype.mutate = function(rate, strength) {
    capturedRate = rate;
    capturedStrength = strength;
  };

  try {
    // Normal elite gen (< 3 stagnant): mutRate = 0.035, mutStrength = 0.08
    agent._recordEvaluationEpisode(100);
    agent._recordEvaluationEpisode(100);
    agent._recordEvaluationEpisode(100);
    assert.equal(capturedRate, 0.035, 'annealed normal rate must be 0.035');
    assert.equal(capturedStrength, 0.08, 'annealed normal strength must be 0.08');

    // Stagnant elite gen (stagnantGens >= 3): mutRate = 0.06, mutStrength = 0.14
    agent.stagnantGens = 3;
    agent._recordEvaluationEpisode(100);
    agent._recordEvaluationEpisode(100);
    agent._recordEvaluationEpisode(100);
    assert.equal(capturedRate, 0.06, 'annealed stagnant rate must be 0.06');
    assert.equal(capturedStrength, 0.14, 'annealed stagnant strength must be 0.14');
  } finally {
    harness.NeuralNetwork.prototype.mutate = originalMutate;
  }
});

test('offset arena wall raycasting: detects wall at ~300 units in offset coordinates (grd=21600)', () => {
  const clock = { t: 0, now() { return this.t; } };
  const harness = loadAgent({ storage: new MemoryStorage(), clock, window: { grd: 21600 } });
  const agent = harness.newAgent();
  const snake = { id: 'me', xx: 42900, yy: 21600, eang: 0, sc: 1.0, sp: 5.9, pts: [] };
  agent._extractInputs(snake, [], [], [], 0.0167, 1000);
  const ray8 = agent.activeRays[8];
  assert.equal(ray8.hitType, 'wall');
  assert.ok(Math.abs(ray8.dist - 300) < 5, `expected ray8 dist ~300, got ${ray8.dist}`);
});

test('coil direction latching: entering COIL with rightClearance > leftClearance latches coilSign = -1', () => {
  const { agent } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.2, sp: 5.9, pts: [] };

  // Frame 1: rightClearance > leftClearance triggers coilSign = -1
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    inputs[29] = 0.8; // pinched flank threat
    inputs[30] = 0.8;
    return {
      inputs,
      minObstacleDist: 400,
      frontObstacleDist: 400,
      leftClearance: 100,
      rightClearance: 250, // right clearance is safer -> turn right (-1)
      forwardObstacleDanger: false,
      normWallDist: 0.1,
      diffCenterAngle: 0,
      headRadius: 18,
      currentAng: 0,
      currentScore: 500,
      cutOffThreat: null,
      attackCandidate: null
    };
  };

  const decision1 = agent.evaluate(snake, [], [], []);
  assert.equal(decision1.stance, 'COIL');
  assert.equal(agent.coilSign, -1, 'must latch coilSign = -1 based on clearance');

  // Frame 2: clearances invert, but coilSign must stay latched at -1
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    inputs[29] = 0.8;
    inputs[30] = 0.8;
    return {
      inputs,
      minObstacleDist: 400,
      frontObstacleDist: 400,
      leftClearance: 300, // inverted clearance
      rightClearance: 50,
      forwardObstacleDanger: false,
      normWallDist: 0.1,
      diffCenterAngle: 0,
      headRadius: 18,
      currentAng: 0,
      currentScore: 500,
      cutOffThreat: null,
      attackCandidate: null
    };
  };

  const decision2 = agent.evaluate(snake, [], [], []);
  assert.equal(decision2.stance, 'COIL');
  assert.equal(agent.coilSign, -1, 'coilSign must remain latched across consecutive frames in COIL');

  // Frame 3: exiting COIL resets coilSign to 0
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 400,
      frontObstacleDist: 400,
      leftClearance: 300,
      rightClearance: 300,
      forwardObstacleDanger: false,
      normWallDist: 0.1,
      diffCenterAngle: 0,
      headRadius: 18,
      currentAng: 0,
      currentScore: 500,
      cutOffThreat: null,
      attackCandidate: null
    };
  };

  const decision3 = agent.evaluate(snake, [], [], []);
  assert.equal(decision3.stance, 'FORAGE');
  assert.equal(agent.coilSign, 0, 'coilSign must reset to 0 upon exiting COIL');
});

test('attack curl scaling: crossingCurl is 0 at long range and ramps up near intercept point', () => {
  const { agent } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, pts: [] };
  const headRadius = 14;

  const runAttack = (distToIntercept) => {
    agent.prevTurnEma = 0;
    agent.boostFuel = 1.0;
    agent.boostCooldown = 0;
    agent._extractInputs = () => {
      const inputs = new Float32Array(32);
      inputs[8] = 0.1; // forward ray clear
      return {
        inputs,
        minObstacleDist: 500,
        frontObstacleDist: 500,
        leftClearance: 200,
        rightClearance: 200,
        forwardObstacleDanger: false,
        normWallDist: 0.1,
        diffCenterAngle: 0,
        headRadius,
        currentAng: 0,
        currentScore: 200,
        cutOffThreat: null,
        attackCandidate: {
          interceptX: 100,
          interceptY: 0, // attackAngle = 0, so attackError = 0
          distToIntercept,
          side: 1 // positive curl
        }
      };
    };
    agent.evaluate(snake, [], [], []);
    // smoothedTurn = turnDelta * 0.7 + prevTurnEma * 0.3 = turnDelta * 0.7 (since prevTurnEma = 0)
    // targetAngle = currentAng + smoothedTurn * 1.85 = turnDelta * 0.7 * 1.85
    // turnDelta = targetAngle / (0.7 * 1.85)
    return agent.targetAngle / (0.7 * 1.85);
  };

  // At long range (dist > headRadius * 8 = 112), crossingCurl should be 0
  const turnDeltaLongRange = runAttack(headRadius * 10);
  assert.equal(turnDeltaLongRange, 0, 'crossingCurl must be 0 when far from intercept point');

  // At close range (dist = headRadius * 3 = 42), closeInFactor is 1.0, crossingCurl should be 0.22
  const turnDeltaCloseRange = runAttack(headRadius * 3);
  assert.ok(Math.abs(turnDeltaCloseRange - 0.22) < 0.01, `crossingCurl should be ~0.22 at close range, got ${turnDeltaCloseRange}`);

  // At medium range (dist = headRadius * 5.5 = 77), closeInFactor is 0.5, crossingCurl should be 0.11
  const turnDeltaMidRange = runAttack(headRadius * 5.5);
  assert.ok(Math.abs(turnDeltaMidRange - 0.11) < 0.01, `crossingCurl should scale smoothly at mid-range, got ${turnDeltaMidRange}`);
});

test('CTRA curvilinear trajectory arc: bends intercept point inward when enemy snake is turning', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.25, sp: 5.9, sct: 250, pts: [] };
  agent.previousScore = 250;
  agent.peakLength = 250;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // Frame 1: Enemy at (180, 50) heading 0 rad
  const enemiesF1 = [{
    id: 'e_curver',
    xx: 180,
    yy: 50,
    eang: 0.0,
    sp: 5.0,
    sc: 0.8,
    dead: false,
    pts: []
  }];
  clock.t = 1000;
  agent.evaluate(snake, enemiesF1, [], []);

  // Frame 2: 100ms later, enemy moves and turns clockwise (ang becomes 0.25 rad, omega = 2.5 rad/s)
  clock.t = 1100;
  const enemiesF2 = [{
    id: 'e_curver',
    xx: 205,
    yy: 56,
    eang: 0.25,
    sp: 5.0,
    sc: 0.8,
    dead: false,
    pts: []
  }];
  const decision = agent.evaluate(snake, enemiesF2, [], []);

  assert.equal(decision.stance, 'ATTACK');
  assert.ok(agent.attackTarget, 'attackTarget must be set');
  // With positive omega (turning right towards +y), curvilinear CTRA interceptY should bend further into +y
  assert.ok(agent.attackTarget.interceptY > enemiesF2[0].yy, `interceptY (${agent.attackTarget.interceptY}) should lead ahead along the turning arc`);
});

// ---------------------------------------------------------------------------
// 7. Apex Tactical Upgrades: Emergency Escape Sprint, Constrictor, Corpse Vacuum
// ---------------------------------------------------------------------------
test('emergency escape sprint: boosts forward into open space when threatened from flank without head-on threat', () => {
  const { agent } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 100, pts: [] };
  agent.previousScore = 100;
  agent.peakLength = 100;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // Simulate DEFEND stance triggered by moderate flank pursuit (threatScore = 0.6) with clear forward exit
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 100,
      frontObstacleDist: 100,
      leftClearance: 100,
      rightClearance: 50,
      forwardObstacleDanger: 0.1,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 100,
      cutOffThreat: { enemyId: 'e1', cutSide: 1, dist: 200, threatScore: 0.6 },
      attackCandidate: null,
      constrictCandidate: null
    };
  };

  const decision = agent.evaluate(snake, [], [], []);
  assert.equal(decision.stance, 'DEFEND', 'should enter DEFEND stance');
  assert.equal(decision.isBoosting, true, 'must fire Emergency Escape Sprint into open space');
});

test('offensive boa constrictor: enters CONSTRICT stance to orbit and trap a smaller enemy snake', () => {
  const { agent } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.5, sp: 5.9, sct: 800, pts: [] };
  agent.previousScore = 800;
  agent.peakLength = 800;

  // Smaller enemy alongside left flank
  const enemies = [{
    id: 'e_prey',
    xx: 50,
    yy: 150,
    eang: 0,
    sp: 5.0,
    sc: 0.7, // < 1.5 * 0.65 = 0.975
    sct: 100, // < 800 * 0.45 = 360
    dead: false,
    pts: []
  }];

  const decision = agent.evaluate(snake, enemies, [], []);
  assert.equal(decision.stance, 'CONSTRICT', 'should enter CONSTRICT stance on vulnerable smaller enemy');
  assert.ok(agent.constrictTarget, 'constrictTarget must be set');
  assert.equal(agent.constrictTarget.side, 1, 'enemy is on left side');
});

test('post-kill corpse vacuum: activates VACUUM stance and prioritizes corpse loot after confirmed kill', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 100, pts: [] };
  agent.previousScore = 100;
  agent.peakLength = 100;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // Trigger confirmed kill by simulating an enemy tracked nearby that vanishes
  agent.enemyTracker.set('e_doomed', {
    xx: 100,
    yy: 50,
    ang: 0,
    vx: 0,
    vy: 0,
    omega: 0,
    timestamp: clock.t,
    wasClose: true,
    dist: 120
  });

  // Enemy vanishes on next frame
  clock.t += 50;
  const foods = [
    { xx: 100, yy: 50, sz: 12, eaten: false, dying: false } // corpse drop
  ];
  const decision = agent.evaluate(snake, [], foods, []);

  assert.equal(decision.stance, 'VACUUM', 'should enter VACUUM stance after confirmed kill');
  assert.equal(agent.kills, 1, 'confirmed kill recorded');
  assert.ok(decision.isBoosting, 'should rush to vacuum corpse loot');
});

// ---------------------------------------------------------------------------
// 6. relative mass ratio & minkowski dynamic clearance engine
// ---------------------------------------------------------------------------
test('minkowski danger clearance: expands avoidance buffer and triggers DEFEND when facing a behemoth enemy', () => {
  const { agent, clock } = makeAgent(1);
  // Small player snake: sc = 1.0 (headRadius = 14)
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 80, pts: [] };
  // Giant behemoth snake placed 125 units ahead: sc = 4.0 (threatRadius = 56)
  // Minkowski dangerDistance = (14 + 56) * 2.25 + 6 = 163.5 > 125!
  // Old formula was: 14 * 4.5 + 6 = 69 < 125 (would have had 0 urgency!)
  const enemies = [{
    id: 'e_behemoth',
    xx: 125,
    yy: 0,
    eang: Math.PI,
    sp: 5.5,
    sc: 4.0,
    dead: false,
    pts: [{ xx: 125, yy: 0 }]
  }];

  const decision = agent.evaluate(snake, enemies, [], []);
  assert.equal(decision.stance, 'DEFEND', 'should enter DEFEND stance against looming behemoth');
  assert.ok(decision.massRatio >= 3.5, `massRatio should be ~4.0, got ${decision.massRatio}`);
  assert.equal(decision.isBoosting, false, 'must not boost forward into a behemoth');
});

test('apex dominance: small prey does not trigger panic evasion when passing at medium distance', () => {
  const { agent, clock } = makeAgent(1);
  // Giant apex player snake: sc = 3.5 (headRadius = 49)
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 3.5, sp: 5.9, sct: 2000, pts: [] };
  // Tiny baby snake 190 units ahead: sc = 0.5 (threatRadius = 7)
  // Minkowski dangerDistance = (49 + 7) * 2.25 + 6 = 132 < 190 -> dangerUrgency = 0!
  // Old formula was: 49 * 4.5 + 6 = 226.5 > 190 (would have panicked with false danger!)
  const enemies = [{
    id: 'e_baby',
    xx: 190,
    yy: 0,
    eang: 0,
    sp: 5.0,
    sc: 0.5,
    dead: false,
    pts: [{ xx: 190, yy: 0 }]
  }];

  const decision = agent.evaluate(snake, enemies, [], []);
  assert.notEqual(decision.stance, 'DEFEND', 'giant apex snake should not panic-defend against distant baby worm');
  assert.ok(decision.massRatio <= 0.2, `massRatio should be small (~0.14), got ${decision.massRatio}`);
});

test('behemoth strike veto: strictly forbids ATTACK stance against a larger opponent', () => {
  const { agent, clock } = makeAgent(1);
  // Player with high score and full boost
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 300, pts: [] };
  agent.previousScore = 300;
  agent.peakLength = 300;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // Larger opponent on flank with massRatio > 1.5
  const enemies = [{
    id: 'e_large',
    xx: 150,
    yy: 60,
    eang: 0.1,
    sp: 4.5,
    sc: 1.8, // 1.8 > 1.0 * 1.5
    dead: false,
    pts: []
  }];

  const decision = agent.evaluate(snake, enemies, [], []);
  assert.notEqual(decision.stance, 'ATTACK', 'must never attempt attack strike against a behemoth threat');
});

test('chase pellet lead interception: projects intercept coordinates and sprints to catch fleeing prey', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 150, pts: [] };
  agent.previousScore = 150;
  agent.peakLength = 150;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // Fleeing prey moving horizontally at speed 7.5
  const preys = [{
    xx: 240,
    yy: 0,
    sp: 7.5,
    ang: 0,
    sz: 25,
    eaten: false,
    dying: false
  }];

  const decision = agent.evaluate(snake, [], [], preys);
  assert.equal(decision.isBoosting, true, 'must sprint to capture high-value fleeing chase pellet');
  assert.ok(agent.detectedFoods.length > 0, 'must detect chase pellet');
  const detected = agent.detectedFoods[0];
  assert.ok(detected.isPrey, 'must be identified as prey');
  // Projected intercept must lead ahead of the raw position (dx > 240)
  assert.ok(detected.dx > 240, `projected intercept (${detected.dx}) must lead ahead of raw position (240)`);
});

test('spiral constriction tightening: tightens inward orbit angle as constriction duration advances', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.5, sp: 5.9, sct: 800, pts: [] };
  agent.previousScore = 800;
  agent.peakLength = 800;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  const enemies = [{
    id: 'e_prey',
    xx: 50,
    yy: 150,
    eang: 0,
    sp: 5.0,
    sc: 0.7,
    sct: 100,
    dead: false,
    pts: []
  }];

  // Initial evaluation at t = 0
  const d1 = agent.evaluate(snake, enemies, [], []);
  assert.equal(d1.stance, 'CONSTRICT');

  // Advance time during active constriction by 1.5s
  clock.t += 1500;
  const d2 = agent.evaluate(snake, enemies, [], []);
  assert.equal(d2.stance, 'CONSTRICT');
  assert.ok(agent.constrictDuration > 0.05, 'constrictDuration should accumulate');
  // Spiral tightening pulls the head further inward toward the target
  assert.ok(Math.abs(d2.targetAngle) !== Math.abs(d1.targetAngle), 'targetAngle should shift as orbit tightens');
});

test('third-party perimeter guard: vetoes CONSTRICT stance when an outer predator is roaming nearby', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.5, sp: 5.9, sct: 800, pts: [] };
  agent.previousScore = 800;
  agent.peakLength = 800;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // Smaller prey on left flank (would normally trigger CONSTRICT)
  // PLUS a larger predator roaming nearby (xx: 200, yy: -150, sc: 2.2)
  const enemies = [
    {
      id: 'e_prey',
      xx: 50,
      yy: 150,
      eang: 0,
      sp: 5.0,
      sc: 0.7,
      sct: 100,
      dead: false,
      pts: []
    },
    {
      id: 'e_predator',
      xx: 200,
      yy: -150,
      eang: Math.PI,
      sp: 5.5,
      sc: 2.2, // > 1.5 * 1.15 = 1.725
      dead: false,
      pts: []
    }
  ];

  const decision = agent.evaluate(snake, enemies, [], []);
  assert.notEqual(decision.stance, 'CONSTRICT', 'must veto CONSTRICT stance when an outer predator is roaming nearby');
});

test('angular boost gating: cuts boost during sharp turns to prevent centrifugal collisions', () => {
  const { agent, clock } = makeAgent(1);
  // High score, plenty of stamina
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 200, pts: [] };
  agent.previousScore = 200;
  agent.peakLength = 200;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // High-value food pellet located at a sharp angle (error > 0.50 rad)
  const foods = [
    { xx: 150, yy: 120, sz: 10, eaten: false, dying: false } // Angle ~38 deg
  ];

  const decision = agent.evaluate(snake, [], foods, []);
  // Food target is at sharp angle, so angular boost gating will veto boosting until aligned
  assert.equal(decision.isBoosting, false, 'must not boost while initiating a sharp turn');
});

// ---------------------------------------------------------------------------
// 8. Semantic Object Radar & Shield Defense Engine (v5.8.0)
// ---------------------------------------------------------------------------
test('semantic object radar: dampens danger urgency when forward obstacle is own body shield (\'self\')', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 100, pts: [] };

  // Case A: Forward obstacle is an enemy head at distance 45
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 45,
      frontObstacleDist: 45,
      frontThreatScale: 1.0,
      nearestThreatScale: 1.0,
      leftClearance: 100,
      rightClearance: 100,
      forwardObstacleDanger: true,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 100,
      cutOffThreat: null,
      attackCandidate: null,
      constrictCandidate: null,
      closestObstacleType: 'head',
      frontObstacleType: 'head',
      selfShieldSide: 0
    };
  };

  const decisionEnemy = agent.evaluate(snake, [], [], []);
  assert.equal(decisionEnemy.stance, 'DEFEND', 'enemy head right ahead must trigger DEFEND');
  assert.equal(decisionEnemy.frontObstacleType, 'head');

  // Case B: Same distance obstacle is our own body shield ('self')
  // In Slither.io, own-body collisions cause ZERO damage.
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 45,
      frontObstacleDist: 45,
      frontThreatScale: 1.0,
      nearestThreatScale: 1.0,
      leftClearance: 100,
      rightClearance: 100,
      forwardObstacleDanger: false,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 100,
      cutOffThreat: null,
      attackCandidate: null,
      constrictCandidate: null,
      closestObstacleType: 'self',
      frontObstacleType: 'self',
      selfShieldSide: 0
    };
  };

  const decisionSelf = agent.evaluate(snake, [], [], []);
  assert.notEqual(decisionSelf.stance, 'DEFEND', 'own body shield must dampen danger urgency by 75% and not panic-defend');
  assert.equal(decisionSelf.frontObstacleType, 'self');
});

test('pocket defense: banks toward own body shield when threatened by enemy head', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 100, pts: [] };

  // Head-on threat with our impenetrable body shield flanking our left side (selfShieldSide = 1)
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 35,
      frontObstacleDist: 35,
      frontThreatScale: 1.0,
      nearestThreatScale: 1.0,
      leftClearance: 50,
      rightClearance: 100, // Even if right has more generic clearance, left has our body shield!
      forwardObstacleDanger: true,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 100,
      cutOffThreat: null,
      attackCandidate: null,
      constrictCandidate: null,
      closestObstacleType: 'head',
      frontObstacleType: 'head',
      selfShieldSide: 1 // Left flank is protected by our own body
    };
  };

  const decision = agent.evaluate(snake, [], [], []);
  assert.equal(decision.stance, 'DEFEND', 'must enter DEFEND stance');
  // In DEFEND, when frontObstacleType === 'head' and selfShieldSide !== 0,
  // agent executes Pocket Defense by steering into its own shield (+1)
  assert.ok(decision.targetAngle > 0, `must steer toward left body shield (+angle), got targetAngle: ${decision.targetAngle}`);
});

// ---------------------------------------------------------------------------
// 9. Grandmaster Combat & Failure-Trauma Filter Engine (v5.9.0)
// ---------------------------------------------------------------------------
test('anti-encirclement slit breach: detects open exit slit and sprints to breach before enemy loop closes', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 400, pts: [] };
  agent.previousScore = 400;
  agent.peakLength = 400;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // A larger enemy wraps around our flanks (pinched and surrounded),
  // which would normally command COIL stance. But an exit slit is open ahead at angle 0!
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    inputs[29] = 0.85; // pinched left
    inputs[30] = 0.85; // pinched right
    inputs[31] = 0.50; // surrounded
    return {
      inputs,
      minObstacleDist: 40,
      frontObstacleDist: 180, // forward exit slit is open
      frontThreatScale: 2.0,
      nearestThreatScale: 2.0,
      leftClearance: 40,
      rightClearance: 40,
      forwardObstacleDanger: false,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 400,
      cutOffThreat: null,
      attackCandidate: null,
      constrictCandidate: null,
      closestObstacleType: 'body',
      frontObstacleType: 'none',
      selfShieldSide: 0,
      encirclementEscape: {
        threatRays: 9,
        slitDist: 180,
        slitAngle: 0, // straight ahead exit
        isOpen: true
      }
    };
  };

  const decision = agent.evaluate(snake, [], [], []);
  assert.equal(decision.stance, 'DEFEND', 'must veto COIL and enter DEFEND for active slit breach');
  assert.equal(decision.isBoosting, true, 'must fire Emergency Slit Breach Sprint through the open gap');
  assert.ok(Math.abs(decision.targetAngle) < 0.35, `must aim toward exit slit (0 rad), got ${decision.targetAngle}`);
});

test('failure-trauma filter: purges final quarantined crash frames upon human teacher death', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 100, pts: [] };

  // Feed 25 frames of human driving into Teacher Mode
  for (let i = 0; i < 25; i++) {
    clock.t += 16;
    agent.learnFromHuman(snake, [], [], [], 100, 50, false);
  }

  // Quarantined frames must be held in FIFO buffer awaiting survival verification
  assert.equal(agent.teacherQuarantineBuffer.length, 25, 'must hold 25 unverified frames in quarantine');
  assert.equal(agent.lifeTeacherSamples, 25, 'lesson counter tracks received tuition');

  // Human makes a fatal mistake and dies
  agent.onDeath(true);

  // Failure-Trauma Filter must incinerate the fatal crash frames immediately
  assert.equal(agent.teacherQuarantineBuffer.length, 0, 'quarantined death frames must be purged from memory on death');
});

test('kinetic enemy boost clearance: expands Minkowski danger buffer when an approaching enemy is boosting', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 80, pts: [] };

  // Enemy placed 125 units ahead
  // When cruising at sp = 5.5, dangerDistance = (14 + 14) * 2.25 + 6 + 0 = 69 < 125 -> no panic
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 125,
      frontObstacleDist: 125,
      frontThreatScale: 1.0,
      nearestThreatScale: 1.0,
      frontThreatSpeed: 5.5, // cruising
      nearestThreatSpeed: 5.5,
      leftClearance: 100,
      rightClearance: 100,
      forwardObstacleDanger: false,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 80,
      cutOffThreat: null,
      attackCandidate: null,
      constrictCandidate: null,
      closestObstacleType: 'head',
      frontObstacleType: 'head',
      selfShieldSide: 0,
      encirclementEscape: null
    };
  };

  const decisionCruise = agent.evaluate(snake, [], [], []);
  assert.notEqual(decisionCruise.stance, 'DEFEND', 'cruising enemy 125 units away should not trigger emergency DEFEND');

  // When enemy is boosting at sp = 13.0, enemySpeedBuffer = (13.0 - 5.5) * 18 = 135!
  // Total dangerDistance = 69 + 135 = 204 > 125 -> dangerUrgency spikes up!
  agent._extractInputs = () => {
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 125,
      frontObstacleDist: 125,
      frontThreatScale: 1.0,
      nearestThreatScale: 1.0,
      frontThreatSpeed: 13.0, // boosting predator!
      nearestThreatSpeed: 13.0,
      leftClearance: 100,
      rightClearance: 100,
      forwardObstacleDanger: true,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 80,
      cutOffThreat: null,
      attackCandidate: null,
      constrictCandidate: null,
      closestObstacleType: 'head',
      frontObstacleType: 'head',
      selfShieldSide: 0,
      encirclementEscape: null
    };
  };

  const decisionBoost = agent.evaluate(snake, [], [], []);
  assert.equal(decisionBoost.stance, 'DEFEND', 'high-speed boosting enemy must trigger proactive DEFEND evasion');
});

test('slipstream wake drafting: boosts food target priority and sprints to harvest boosting enemy exhaust trail', () => {
  const { agent, clock } = makeAgent(1);
  const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 200, pts: [] };
  agent.previousScore = 200;
  agent.peakLength = 200;
  agent.boostFuel = 1.0;
  agent.boostCooldown = 0;

  // Boosting enemy ahead at distance 250, dropping boost exhaust pellets at distance 200
  agent._extractInputs = () => {
    agent.detectedFoods = [
      { dx: 200, dy: 10, dist: 200, sz: 4, isPrey: false }
    ];
    const inputs = new Float32Array(32);
    return {
      inputs,
      minObstacleDist: 250,
      frontObstacleDist: 250,
      frontThreatScale: 1.0,
      nearestThreatScale: 1.0,
      frontThreatSpeed: 11.0,
      nearestThreatSpeed: 11.0,
      leftClearance: 100,
      rightClearance: 100,
      forwardObstacleDanger: false,
      normWallDist: 0.2,
      diffCenterAngle: 0,
      headRadius: 14,
      currentAng: 0,
      currentScore: 200,
      cutOffThreat: null,
      attackCandidate: null,
      constrictCandidate: null,
      closestObstacleType: 'none',
      frontObstacleType: 'none',
      selfShieldSide: 0,
      encirclementEscape: null,
      slipstreamTarget: { enemyId: 'e_sprinter', dist: 250, speed: 11.0, side: 1 }
    };
  };

  // Pellet in slipstream wake (dist = 200, sz = 4 - would normally be low priority without slipstream)
  const foods = [
    { xx: 200, yy: 10, sz: 4, eaten: false, dying: false }
  ];

  const decision = agent.evaluate(snake, [], foods, []);
  assert.equal(decision.isBoosting, true, 'must activate slipstream drafting sprint to harvest enemy boost wake');
});
