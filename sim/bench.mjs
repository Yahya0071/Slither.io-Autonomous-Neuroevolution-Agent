#!/usr/bin/env node
/**
 * Behaviour benchmark: run the simulator across a fixed set of scenarios and
 * print the metrics that matter for the two reported symptoms - boosting too
 * much (length loss) and not following food.
 *
 * Usage: node sim/bench.mjs [extra label]
 */
import { simulate } from './snake-sim.mjs';

const SCENARIOS = [
  { label: 'scattered food, no enemies', opts: { foodLayout: 'uniform', foodCount: 400, seed: 7 } },
  { label: 'food clumped straight ahead', opts: { foodLayout: 'ahead', foodCount: 300, seed: 11 } },
  { label: 'food clumped 90deg left', opts: { foodLayout: 'left', foodCount: 300, seed: 13 } },
  { label: 'no food at all', opts: { foodLayout: 'none', foodCount: 0, seed: 17 } },
  { label: 'no food, ringed by 8 snakes', opts: { foodLayout: 'none', foodCount: 0, enemies: 8, seed: 19 } },
  { label: 'scattered food + 8 snakes', opts: { foodLayout: 'uniform', foodCount: 400, enemies: 8, seed: 23 } },
];

const pad = (s, n) => String(s).padEnd(n);
const padL = (s, n) => String(s).padStart(n);

const rows = [];
for (const { label, opts } of SCENARIOS) {
  // average three seeds so a single lucky layout cannot dominate
  const runs = [0, 1, 2].map((k) => simulate({ ...opts, seed: opts.seed + k * 31 }));
  const avg = (key) => runs.reduce((s, r) => s + r[key], 0) / runs.length;
  rows.push({
    label,
    pellets: avg('pelletsPerMinute'),
    score: avg('scoreDelta'),
    boost: avg('boostDuty') * 100,
    pointed: avg('pointedAtFood'),
    dist: avg('meanNearestFoodDist'),
    eff: avg('pathEfficiency'),
  });
}

const header = `${pad('scenario', 32)}${padL('pellets/min', 12)}${padL('score/min', 11)}${padL('boost %', 9)}${padL('aimed@food', 12)}${padL('nearest', 9)}${padL('path eff', 10)}`;
console.log('\n' + header);
console.log('-'.repeat(header.length));
for (const r of rows) {
  console.log(
    pad(r.label, 32) +
      padL(r.pellets.toFixed(1), 12) +
      padL(r.score.toFixed(1), 11) +
      padL(r.boost.toFixed(1), 9) +
      padL(r.pointed === null ? 'n/a' : (r.pointed * 100).toFixed(0) + '%', 12) +
      padL(r.dist === null ? 'n/a' : r.dist.toFixed(0), 9) +
      padL(r.eff.toFixed(2), 10),
  );
}
console.log('\nboost %     = share of frames the agent holds boost (length is bleeding when this is high)');
console.log('aimed@food  = share of frames the nearest pellet is within +/-60deg of the head');
console.log('nearest     = mean distance to the nearest pellet (world units)');
console.log('path eff    = net displacement / distance travelled (1.0 = straight, ~0 = circling)\n');
