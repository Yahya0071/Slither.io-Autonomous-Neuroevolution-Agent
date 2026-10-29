import { makeAgent } from './snake-sim.mjs';

const { agent, clock } = makeAgent(7);
const foods = Array.from({ length: 40 }, (_, i) => ({ xx: 300 + (i % 8) * 40, yy: -100 + Math.floor(i / 8) * 50, sz: 4 }));
const snake = { id: 'me', xx: 0, yy: 0, eang: 0, sc: 1 + 100 / 15000, sp: 5.9, pts: [], dead: false };

console.log('frame  targetAngle  eang    turnDelta  boost  x       y');
for (let f = 0; f < 12; f++) {
  clock.t += 16.67;
  const d = agent.evaluate(snake, [], foods, []);
  const diff = Math.atan2(Math.sin(d.targetAngle - snake.eang), Math.cos(d.targetAngle - snake.eang));
  console.log(
    String(f).padStart(5),
    d.targetAngle.toFixed(3).padStart(11),
    snake.eang.toFixed(3).padStart(7),
    diff.toFixed(3).padStart(9),
    String(d.isBoosting).padStart(6),
    snake.xx.toFixed(2).padStart(8),
    snake.yy.toFixed(2).padStart(8),
  );
  snake.eang += Math.max(-0.075, Math.min(0.075, diff));
  snake.xx += Math.cos(snake.eang) * 5.9 / 60;
  snake.yy += Math.sin(snake.eang) * 5.9 / 60;
}

console.log('\ninputs that drive steering:');
const st = agent._extractInputs(snake, [], foods, [], 0.0167, clock.t);
console.log('rays 0..15 :', Array.from(st.inputs.slice(0, 16), (v) => v.toFixed(2)).join(' '));
console.log('food 16..23:', Array.from(st.inputs.slice(16, 24), (v) => v.toFixed(3)).join(' '));
console.log('geom 24..27:', Array.from(st.inputs.slice(24, 28), (v) => v.toFixed(3)).join(' '));
console.log('mem 28..31  :', Array.from(st.inputs.slice(28, 32), (v) => v.toFixed(3)).join(' '));
const out = agent.brain.predict(st.inputs);
console.log('network out : steer=' + out[0].toFixed(4) + '  boost=' + out[1].toFixed(4));
