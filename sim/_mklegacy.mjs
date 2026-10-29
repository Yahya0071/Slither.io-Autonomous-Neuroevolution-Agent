// One-off: rebuild the pre-fix agent by reversing the four behavioural changes,
// so before/after can be measured on the same yardstick. Deleted after use.
import { readFileSync, writeFileSync } from 'node:fs';

const src = readFileSync('console_inject.js', 'utf8');
const pairs = [
  // 1. food radar was normalised by 40 (effectively invisible)
  ['Math.min(1.0, this.foodSectors[f] / 8.0)', 'Math.min(1.0, this.foodSectors[f] / 40)'],
  // 2. centre-seek was ungated and degenerate at the origin
  ['const wallPressure = Math.max(0, Math.min(1, (normWallDist - 0.55) / 0.35));', 'const wallPressure = 1;'],
  ['const angleToCenter = distFromCenter > 1 ? Math.atan2(-headY, -headX) : currentAng;', 'const angleToCenter = Math.atan2(-headY, -headX);'],
  // 3. collision urgency used the 360-degree minimum (own tail included)
  ['const dangerSource = Number.isFinite(frontObstacleDist) ? frontObstacleDist : minObstacleDist;', 'const dangerSource = minObstacleDist;'],
  // 4. no explicit foraging term
  ['if (foodTarget) {', 'if (false && foodTarget) {'],
  // 5. no boost governor (old emergency-escape bypass)
  ['this.boostBurstSeconds = 1.2;', 'this.boostBurstSeconds = 2.0;'],
  ['this.boostRestSeconds = 2.0;', 'this.boostRestSeconds = 1.4;'],
  ['this.boostBurnSeconds = 1.6;', 'this.boostBurnSeconds = 1e9;'],
  [
    `const shouldBoost = maySprint && boostProb > 0.5;`,
    `const emergencyEscape = forwardClear && (this.leftThreatEma > 0.75 || this.rightThreatEma > 0.75) && currentScore > 40;
      const shouldBoost = forwardClear && ((boostProb > 0.72 && richFoodAhead && inputs[31] < 0.5) || emergencyEscape);`,
  ],
];

let out = src;
for (const [from, to] of pairs) {
  if (!out.includes(from)) throw new Error('patch did not apply: ' + from.slice(0, 60));
  out = out.replace(from, to);
}
writeFileSync('_legacy_ref.js', out, 'utf8');
console.log('wrote _legacy_ref.js (' + out.length + ' bytes,', out === src ? 'UNCHANGED' : 'patched', ')');
