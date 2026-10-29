import { makeAgent } from './snake-sim.mjs';
const { agent, clock } = makeAgent(3);
for (const n of [200, 2000, 8000]) {
  // uniform pellets in a 2600-radius disc, snake at origin
  let seed = 42;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const foods = Array.from({ length: n }, () => {
    const r = Math.sqrt(rand()) * 2600, a = rand() * Math.PI * 2;
    return { xx: Math.cos(a) * r, yy: Math.sin(a) * r, sz: 4 };
  });
  const snake = { xx: 0, yy: 0, eang: 0, sc: 1.02, sp: 5.9, pts: [] };
  const st = agent._extractInputs(snake, [], foods, [], 0.0167, clock.t);
  const sectors = Array.from(st.inputs.slice(16, 24), (v) => v.toFixed(3));
  console.log(`${String(n).padStart(5)} pellets -> food sector inputs: ${sectors.join(' ')}  (max ${Math.max(...Array.from(st.inputs.slice(16,24))).toFixed(3)})`);
}
