/**
 * SlitherML - Modular Architecture Index
 * Slither.io Autonomous Neuroevolution & Imitation Learning Agent
 */

export { NeuralNetwork } from './network.js';
export { FlightRecorder } from './flight-recorder.js';
export { SlitherMLAgent, STORAGE_KEY } from './agent.js';
export { SlitherMLController } from './controller.js';
export {
  extractCombatGeometry,
  castProximityRays,
  findEscapeCorridors,
  extractFoodRadar,
  pickFoodTarget
} from './perception.js';
export {
  arbitrateStance,
  computeSteering,
  governBoost
} from './stances.js';
export {
  evaluateRollout,
  breedNextGenome
} from './evolution.js';
