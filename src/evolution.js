/**
 * Evolutionary Engine: Rollout Scoring, Consistency Gate, Plateau Relief,
 * Silver Brain Demotion/Adoption, Uniform Crossover, and Fast Evolutionary Programming Mutation.
 */

import { NeuralNetwork } from './network.js';

export function evaluateRollout({
  episodeFitnesses,
  previousBest,
  allTimePeakFitness,
  stagnantGens,
  candidatePending,
  candidateSource,
  isAutonomousMutant,
  inputCount = 32
}) {
  const sorted = [...episodeFitnesses].sort((a, b) => b - a);
  const aggregateFitness = (sorted[0] + sorted[1]) / 2;
  const meanFitness = episodeFitnesses.reduce((a, b) => a + b, 0) / episodeFitnesses.length;

  const isTeacherCandidate = candidatePending && candidateSource === 'teacher';
  const teacherWorthyOfChampBlend = isTeacherCandidate && (aggregateFitness >= previousBest * 0.35 || aggregateFitness >= 5000);
  const teacherWorthyOfBreedingBlend = isTeacherCandidate && !teacherWorthyOfChampBlend && aggregateFitness <= previousBest;

  const passesConsistencyGate = sorted[1] >= previousBest * 0.40;
  const mutantWorthyOfBlend = isAutonomousMutant && aggregateFitness <= previousBest && aggregateFitness >= previousBest * 0.40;

  const isProdigyRun = isAutonomousMutant && !mutantWorthyOfBlend && aggregateFitness <= previousBest &&
    sorted[0] >= Math.min(22000, (allTimePeakFitness || previousBest) * 0.55);

  const isNewChampion = aggregateFitness > previousBest && passesConsistencyGate;

  let nextBestFitness = previousBest;
  let nextPeakFitness = Math.max(allTimePeakFitness || 0, previousBest);
  let nextStagnantGens = stagnantGens;
  let silverUpdate = null;

  if (isNewChampion) {
    nextStagnantGens = 0;
    nextBestFitness = aggregateFitness;
    nextPeakFitness = Math.max(nextPeakFitness, aggregateFitness);
  } else {
    nextStagnantGens = (stagnantGens || 0) + 1;
    const floorRatio = nextStagnantGens >= 60 ? 0.70 : (nextStagnantGens >= 50 ? 0.75 : 0.85);
    const minBar = nextPeakFitness * floorRatio;
    if (nextStagnantGens >= 15) {
      nextBestFitness = Math.max(minBar, previousBest * 0.990);
    } else if (nextStagnantGens >= 6) {
      nextBestFitness = Math.max(minBar, previousBest * 0.995);
    }

    if ((mutantWorthyOfBlend || isProdigyRun)) {
      silverUpdate = {
        fitness: aggregateFitness,
        source: 'mutant'
      };
    }
  }

  return {
    sorted,
    aggregateFitness,
    meanFitness,
    isNewChampion,
    passesConsistencyGate,
    mutantWorthyOfBlend,
    isProdigyRun,
    teacherWorthyOfChampBlend,
    teacherWorthyOfBreedingBlend,
    nextBestFitness,
    nextPeakFitness,
    nextStagnantGens,
    silverUpdate
  };
}

export function breedNextGenome({
  championBrain,
  silverBrain,
  candidateBrain,
  stagnantGens,
  bestFitness,
  inputCount = 32,
  blendOptions = {}
}) {
  const {
    mutantWorthyOfBlend = false,
    teacherWorthyOfBreedingBlend = false,
    isProdigyRun = false,
    aggregateFitness = 0,
    peakLifeFitness = 0
  } = blendOptions;

  if (!championBrain) {
    const net = new NeuralNetwork(inputCount, [48, 24], 2);
    net.mutate(0.08, 0.20);
    return net;
  }

  const useCrossover = stagnantGens >= 5 && Boolean(silverBrain);
  const baseBrain = useCrossover
    ? NeuralNetwork.crossover(championBrain, silverBrain)
    : NeuralNetwork.fromJSON(championBrain.export(), inputCount);

  if (!baseBrain) return null;

  if ((mutantWorthyOfBlend || teacherWorthyOfBreedingBlend || isProdigyRun) && candidateBrain) {
    let blendRatio = 0.15;
    if (isProdigyRun) {
      blendRatio = 0.20;
    }
    const keepRatio = 1.0 - blendRatio;
    for (let layer = 0; layer < baseBrain.weights.length; layer++) {
      const bW = baseBrain.weights[layer];
      const cW = candidateBrain.weights[layer];
      for (let j = 0; j < bW.length; j++) {
        bW[j] = (bW[j] * keepRatio) + (cW[j] * blendRatio);
      }
      const bB = baseBrain.biases[layer];
      const cB = candidateBrain.biases[layer];
      for (let j = 0; j < bB.length; j++) {
        bB[j] = (bB[j] * keepRatio) + (cB[j] * blendRatio);
      }
    }
    if (baseBrain.biases && baseBrain.biases.length > 0) {
      const baseOutB = baseBrain.biases[baseBrain.biases.length - 1];
      if (baseOutB && baseOutB.length > 0) {
        baseOutB[0] = Math.max(-0.15, Math.min(0.15, baseOutB[0]));
      }
    }
  }

  const isStagnant = stagnantGens >= 3;
  const isDeepStagnant = stagnantGens >= 8;
  const isElite = bestFitness > 30000;

  let mutRate = isDeepStagnant ? 0.18 : (isStagnant ? 0.12 : 0.06);
  let mutStrength = isDeepStagnant ? 0.35 : (isStagnant ? 0.28 : 0.16);

  if (isElite) {
    mutRate = isDeepStagnant ? 0.09 : (isStagnant ? 0.06 : 0.035);
    mutStrength = isDeepStagnant ? 0.20 : (isStagnant ? 0.14 : 0.08);
  }

  baseBrain.mutate(mutRate, mutStrength);
  return baseBrain;
}
