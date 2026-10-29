import { makeAgent } from './snake-sim.mjs';

const { agent, clock } = makeAgent(11);
// food in a +/-30 deg cone ahead
let seed = 11 * 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const foods = Array.from({ length: 300 }, () => {
  const a = (rand() - 0.5) * (Math.PI / 3);
  const d = 200 + rand() * 2400;
  return { xx: Math.cos(a) * d, yy: Math.sin(a) * d, sz: 4 };
});

const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1 + 100 / 15000, sp: 5.9, pts: [], dead: false };
let eaten = 0;

console.log('frame    x       y     eang  nearest detected foodErr danger minObs  ate');
for (let f = 0; f < 90; f++) {
  clock.t += 16.67;
  const d = agent.evaluate(snake, [], foods, []);
  const diff = Math.atan2(Math.sin(d.targetAngle - snake.eang), Math.cos(d.targetAngle - snake.eang));
  snake.eang += Math.max(-0.075, Math.min(0.075, diff));
  snake.xx += Math.cos(snake.eang) * 354 / 60;
  snake.yy += Math.sin(snake.eang) * 354 / 60;


  const headR = 14 * snake.sc;
  let ateThis = 0;
  for (let i = foods.length - 1; i >= 0; i--) {
    if (Math.hypot(foods[i].xx - snake.xx, foods[i].yy - snake.yy) < headR + 12) {
      foods.splice(i, 1); eaten++; ateThis++;
    }
  }

  let nearest = Infinity;
  for (const fd of foods) nearest = Math.min(nearest, Math.hypot(fd.xx - snake.xx, fd.yy - snake.yy));

  if (f % 6 === 0) {
    const st = agent._extractInputs(snake, [], foods, [], 0.0167, clock.t);
    const urg = Math.max(0, Math.min(1, (st.headRadius * 4.5 - st.minObstacleDist) / (st.headRadius * 3)));
    console.log(
      String(f).padStart(5),
      snake.xx.toFixed(0).padStart(7),
      snake.yy.toFixed(0).padStart(7),
      snake.eang.toFixed(2).padStart(6),
      nearest.toFixed(0).padStart(7),
      String(agent.detectedFoods.length).padStart(8),
      (agent.foodTargetError === null ? 'null' : agent.foodTargetError.toFixed(2)).padStart(7),
      urg.toFixed(2).padStart(6),
      st.minObstacleDist.toFixed(0).padStart(6),
      String(ateThis).padStart(4),
    );
  }
}
console.log(`\neaten total: ${eaten} in ${(90 / 60).toFixed(1)}s`);
