import { makeAgent } from './snake-sim.mjs';

// Drive the boost governor with synthetic states so each gate can be isolated.
function duty({ score = 200, peak = 0, frontBlocked = false, congested = false, target = true, seconds = 40 }) {
  const { agent, clock } = makeAgent(1);
  const inputs = new Float32Array(32);
  if (frontBlocked) { inputs[7] = 0.9; inputs[8] = 0.9; inputs[9] = 0.9; }
  if (congested) inputs[31] = 0.9;

  agent._extractInputs = () => ({
    inputs,
    minObstacleDist: 600,
    frontObstacleDist: frontBlocked ? 20 : 600,
    leftClearance: 100,
    rightClearance: 100,
    forwardObstacleDanger: false,
    normWallDist: 0.1,
    diffCenterAngle: 0,
    headRadius: 20,
    currentAng: 0,
    currentScore: score,
  });

  agent.peakLength = peak;
  // a dead-snake drop 200 units straight ahead (value 8 => worth sprinting)
  agent.detectedFoods = target ? [{ dx: 200, dy: 0, dist: 200, sz: 20, isPrey: false }] : [];

  let boosting = 0;
  const frames = seconds * 60;
  for (let f = 0; f < frames; f++) {
    clock.t += 1000 / 60;
    const d = agent.evaluate({ xx: 0, yy: 0, sc: 1 + score / 15000, eang: 0, sp: 5.9 }, [], [], []);
    if (d.isBoosting) boosting++;
  }
  return (boosting / frames) * 100;
}

const cases = [
  { label: 'big drop ahead, healthy snake', args: {} },
  { label: 'no worthwhile target (plain pellets)', args: { target: false } },
  { label: 'score 50 (below the 80 floor)', args: { score: 50 } },
  { label: 'depleted: 200 from a 300 peak', args: { score: 200, peak: 300 } },
  { label: 'front blocked', args: { frontBlocked: true } },
  { label: 'congested traffic', args: { congested: true } },
];

console.log('boost duty cycle under the governor:\n');
for (const c of cases) {
  console.log(`  ${c.label.padEnd(38)} ${duty(c.args).toFixed(1).padStart(6)} %`);
}
console.log('\nmax possible duty = burst(1.2s) / (burst + rest 2.0s) = 37.5% while a target persists');
