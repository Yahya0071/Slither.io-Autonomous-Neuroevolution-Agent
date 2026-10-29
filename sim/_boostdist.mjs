import { makeAgent } from './snake-sim.mjs';
const { agent, clock } = makeAgent(5);
let seed = 99;
const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const foods = Array.from({ length: 1500 }, () => {
  const r = Math.sqrt(rand()) * 2600, a = rand() * Math.PI * 2;
  return { xx: Math.cos(a) * r, yy: Math.sin(a) * r, sz: 4 };
});
// plus a juicy dead-snake drop straight ahead
for (let i = 0; i < 30; i++) foods.push({ xx: 240 + rand() * 60, yy: (rand() - 0.5) * 60, sz: 20 });
const snake = { xx: 0, yy: 0, eang: 0, sc: 1 + 100 / 15000, sp: 5.9, pts: [] };
const vals = [];
for (let f = 0; f < 600; f++) {
  clock.t += 16.67;
  const st = agent._extractInputs(snake, [], foods, [], 0.0167, clock.t);
  vals.push(agent.brain.predict(st.inputs)[1]);
}
vals.sort((a, b) => a - b);
const q = (p) => vals[Math.floor(vals.length * p)].toFixed(3);
console.log('network boost output over 600 dense-food frames:');
console.log(`  min ${q(0.01)}  p25 ${q(0.25)}  median ${q(0.5)}  p75 ${q(0.75)}  p95 ${q(0.95)}  max ${q(0.999)}`);
console.log(`  frames above 0.50: ${(vals.filter((v) => v > 0.5).length / vals.length * 100).toFixed(1)}%`);
console.log(`  frames above 0.35: ${(vals.filter((v) => v > 0.35).length / vals.length * 100).toFixed(1)}%`);
