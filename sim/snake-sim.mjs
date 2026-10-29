/**
 * Headless flight simulator.
 *
 * Models just enough of slither.io to measure the agent's *behaviour*: how much
 * of the time it boosts, whether it points at food, and how much it eats. It is
 * deliberately small (no enemy AI, no server, no rendering) and it is a MODEL,
 * not the real game - absolute numbers are indicative, but A/B comparisons run
 * through the same seeds are a reliable signal.
 *
 * World model:
 *   - the head turns toward the agent's targetAngle at a bounded rate
 *   - motion is at a constant speed, ~1.9x while boosting
 *   - boosting drains score, pellets add score, catch radius is the head radius
 *   - scale is derived exactly as the agent derives it: score = 15000*(sc-1)
 */
import { loadAgent, MemoryStorage } from '../agent-loader.mjs';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function rand() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const wrap = (a) => {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
};

const DEFAULTS = {
  seconds: 60,
  fps: 60,
  seed: 1,
  startScore: 100,
  speed: 354, // world units per SECOND (~5.9 per frame at 60fps)
  boostMultiplier: 1.9,
  maxTurnRate: 4.5, // rad/s
  boostDrainPerSecond: 3, // score units burned per second of boosting
  pelletValue: 1,
  catchRadius: 12,
  foodCount: 400,
  foodRadius: 2600,
  foodLayout: 'uniform', // uniform | ahead | left | none
  enemies: 0,
  enemyRingRadius: 260,
  wallRadius: 16384,
  trailSpacing: 7,
  maxTrail: 260,
};

/** Build a fresh agent wired to a virtual clock so dt is exact and repeatable. */
export function makeAgent(seed = 1) {
  const clock = { t: 0, now() { return this.t; } };
  const harness = loadAgent({ storage: new MemoryStorage(), clock, window: { grd: DEFAULTS.wallRadius } });
  const agent = harness.newAgent();
  agent.resetBrain();
  agent.bestFitness = 0;
  return { agent, clock, harness, seed };
}

function placeFood(rand, opts) {
  const { foodCount, foodRadius, foodLayout } = opts;
  const foods = [];
  for (let i = 0; i < foodCount; i++) {
    if (foodLayout === 'ahead') {
      const angle = (rand() - 0.5) * (Math.PI / 3); // +/-30 deg cone, heading 0
      const dist = 200 + rand() * (foodRadius - 200);
      foods.push({ xx: Math.cos(angle) * dist, yy: Math.sin(angle) * dist, sz: 4 });
    } else if (foodLayout === 'left') {
      const angle = -Math.PI / 2 + (rand() - 0.5) * (Math.PI / 3); // 90 deg to the snake's left
      const dist = 200 + rand() * (foodRadius - 200);
      foods.push({ xx: Math.cos(angle) * dist, yy: Math.sin(angle) * dist, sz: 4 });
    } else {
      const r = Math.sqrt(rand()) * foodRadius;
      const a = rand() * Math.PI * 2;
      foods.push({ xx: Math.cos(a) * r, yy: Math.sin(a) * r, sz: 4 });
    }
  }
  return foods;
}

function placeEnemies(rand, opts) {
  const list = [];
  for (let i = 0; i < opts.enemies; i++) {
    const a = (i / opts.enemies) * Math.PI * 2;
    const r = opts.enemyRingRadius + rand() * 120;
    const cx = Math.cos(a) * r;
    const cy = Math.sin(a) * r;
    const pts = [];
    for (let p = 0; p < 60; p++) {
      pts.push({ xx: cx + Math.cos(a + p * 0.05) * p * 6, yy: cy + Math.sin(a + p * 0.05) * p * 6, dying: false });
    }
    list.push({ id: 'e' + i, xx: cx, yy: cy, sc: 1.4, eang: a, sp: 5.9, pts, dead: false });
  }
  return list;
}

/**
 * Run one scenario and return behaviour metrics.
 * @param {object} [opts] overrides for DEFAULTS, plus `agent` to reuse one.
 */
export function simulate(opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const rand = mulberry32(o.seed);

  const setup = o.agent ? { agent: o.agent, clock: o.clock } : makeAgent(o.seed);
  const { agent, clock } = setup;

  const foods = placeFood(rand, o);
  const enemies = placeEnemies(rand, o);

  const snake = {
    id: 'me',
    xx: 0,
    yy: 0,
    eang: o.startAngle ?? 0,
    sc: 1 + o.startScore / 15000,
    sp: 5.9,
    pts: [],
    dead: false,
  };

  let score = o.startScore;
  let pelletsEaten = 0;
  let boostFrames = 0;
  let frames = 0;
  let travel = 0;
  let pointedAtFoodFrames = 0;
  let foodFrames = 0;
  let nearestFoodDistSum = 0;
  let lastTrail = { xx: 0, yy: 0 };
  const dt = 1 / o.fps;

  for (let f = 0; f < o.seconds * o.fps; f++) {
    clock.t += dt * 1000;
    frames++;

    const decision = agent.evaluate(snake, enemies, foods, []);

    // bounded turn toward the commanded angle
    const diff = wrap(decision.targetAngle - snake.eang);
    const maxStep = o.maxTurnRate * dt;
    snake.eang = wrap(snake.eang + Math.max(-maxStep, Math.min(maxStep, diff)));

    const boosting = Boolean(decision.isBoosting);
    if (boosting) boostFrames++;
    const v = boosting ? o.speed * o.boostMultiplier : o.speed;

    snake.xx += Math.cos(snake.eang) * v * dt;
    snake.yy += Math.sin(snake.eang) * v * dt;
    travel += v * dt;

    // trail (used by the agent's self-collision rays)
    if (Math.hypot(snake.xx - lastTrail.xx, snake.yy - lastTrail.yy) > o.trailSpacing) {
      snake.pts.unshift({ xx: snake.xx, yy: snake.yy, dying: false });
      if (snake.pts.length > o.maxTrail) snake.pts.length = o.maxTrail;
      lastTrail = { xx: snake.xx, yy: snake.yy };
    }

    // eat
    const headRadius = 14 * snake.sc;
    const catchR = headRadius + o.catchRadius;
    for (let i = foods.length - 1; i >= 0; i--) {
      const fd = foods[i];
      if (Math.hypot(fd.xx - snake.xx, fd.yy - snake.yy) < catchR) {
        foods.splice(i, 1);
        score += o.pelletValue;
        pelletsEaten++;
      }
    }

    // boost burns length
    if (boosting) score = Math.max(6, score - o.boostDrainPerSecond * dt);

    snake.sc = 1 + score / 15000;
    snake.sp = boosting ? 11.2 : 5.9;

    // is the nearest pellet roughly in front of the head?
    let nearest = Infinity;
    for (let i = 0; i < foods.length; i++) {
      const d = Math.hypot(foods[i].xx - snake.xx, foods[i].yy - snake.yy);
      if (d < nearest) nearest = d;
    }
    if (nearest < 900) {
      foodFrames++;
      nearestFoodDistSum += nearest;
      let bearing = 0;
      for (let i = 0; i < foods.length; i++) {
        const d = Math.hypot(foods[i].xx - snake.xx, foods[i].yy - snake.yy);
        if (d === nearest) {
          bearing = Math.atan2(foods[i].yy - snake.yy, foods[i].xx - snake.xx);
          break;
        }
      }
      if (Math.abs(wrap(bearing - snake.eang)) < Math.PI / 3) pointedAtFoodFrames++;
    }

    if (Math.hypot(snake.xx, snake.yy) > o.wallRadius * 0.5) {
      snake.dead = true;
      break;
    }
  }

  return {
    seconds: frames / o.fps,
    frames,
    diedAtWall: snake.dead,
    pelletsEaten,
    pelletsPerMinute: pelletsEaten / (frames / o.fps) * 60,
    finalScore: score,
    scoreDelta: score - o.startScore,
    boostDuty: boostFrames / frames,
    travel,
    netDisplacement: Math.hypot(snake.xx, snake.yy),
    pathEfficiency: travel > 0 ? Math.hypot(snake.xx, snake.yy) / travel : 0,
    pointedAtFood: foodFrames > 0 ? pointedAtFoodFrames / foodFrames : null,
    meanNearestFoodDist: foodFrames > 0 ? nearestFoodDistSum / foodFrames : null,
    foodsLeft: foods.length,
  };
}
