/**
 * SlitherMLAgent: Core Neuroevolution & Online Imitation Learning Autonomous Agent
 */

import { NeuralNetwork } from './network.js';
import { FlightRecorder } from './flight-recorder.js';
import {
  extractCombatGeometry,
  castProximityRays,
  findEscapeCorridors,
  extractFoodRadar,
  pickFoodTarget
} from './perception.js';
import {
  arbitrateStance,
  computeSteering,
  governBoost
} from './stances.js';
import {
  evaluateRollout,
  breedNextGenome
} from './evolution.js';

export const STORAGE_KEY = 'slither_ml_champion_weights_v4';

const getPageWindow = () => {
  if (typeof unsafeWindow !== 'undefined' && unsafeWindow) return unsafeWindow;
  if (typeof window !== 'undefined') return window;
  return { grd: 21600 };
};

export class SlitherMLAgent {
  constructor(options = {}) {
    this.flightRecorder = new FlightRecorder();
    this.rayCount = 16;
    this.baseSensorRange = 520;
    this.foodSectorCount = 8;
    this.inputCount = 32;
    this.brain = new NeuralNetwork(this.inputCount, [48, 24], 2);
    this.championBrain = null;
    this.silverBrain = null;
    this.silverFitness = 0;
    this.silverSource = null;

    // Multi-Episode Evaluation Architecture
    this.episodesPerGenome = 3;
    this.currentEpisode = 1;
    this.episodeFitnesses = [];
    this.recentRollouts = [];

    this.candidatePending = false;
    this.candidateSource = null;
    this.abandonCurrentEpisode = false;

    this.generation = 1;
    this.fitness = 0;
    this.bestFitness = 0;
    this.allTimePeakFitness = 0;
    this.peakLength = 0;
    this.previousScore = 0;
    this.survivalTime = 0;
    this.totalFoodEaten = 0;

    // Teacher / Imitation Learning Metrics
    this.teacherSamples = 0;
    this.lifeTeacherSamples = 0;
    this.touchedByHuman = false;

    // Enemy Velocity & Trajectory Tracker
    this.enemyTracker = new Map();
    this.isCenteredWorld = true;

    // Short-Term Tactical Memory Buffers (EMA decay = 0.85)
    this.prevTurnEma = 0;
    this.leftThreatEma = 0;
    this.rightThreatEma = 0;

    this.activeRays = [];
    this.foodSectors = new Float32Array(this.foodSectorCount);
    this.detectedFoods = [];
    this.targetAngle = 0;
    this.isBoosting = false;
    this.boostDuration = 0;
    this.boostCooldown = 0;
    this.foodTargetError = null;

    this.boostFuel = 1;
    this.boostBurstSeconds = 1.2;
    this.boostBurnSeconds = 1.6;
    this.boostRefillSeconds = 4.0;
    this.boostRestSeconds = 2.0;
    this.minBoostScore = 80;
    this.foodSteerEma = 0;
    this.lastDecisionTime = typeof performance !== 'undefined' ? performance.now() : Date.now();

    // Combat & Tactical Stance State Machine
    this.combatStance = 'FORAGE';
    this.attackTarget = null;
    this.cutOffThreat = null;
    this.constrictTarget = null;
    this.constrictDuration = 0;
    this.huntTarget = null;
    this.huntDuration = 0;
    this.aggressionScore = 0;
    this.coilSign = 0;
    this.lastKillEpicenter = null;
    this.vacuumActiveUntil = 0;
    this.massRatio = 1.0;
    this.threatScale = 1.0;
    this.closestObstacleType = 'none';
    this.frontObstacleType = 'none';
    this.selfShieldSide = 0;
    this.encirclementEscape = null;
    this.slipstreamTarget = null;
    this.evasionSign = 0;
    this.evasionLatchTimer = 0;
    this.escapeCorridorAngle = null;
    this.corridorTurnSign = 0;
    this.teacherQuarantineBuffer = [];
    this.teacherQuarantineCapacity = 45;
    this.kills = 0;
    this.stagnantGens = 0;
    this.allTimePeakLength = 0;
    this.lastLoadedTimestamp = 0;
    this._saveCheckpoint = () => this._saveChampionWeights();

    this._loadChampionWeights();
  }

  applySavedState(parsed) {
    if (!parsed || typeof parsed !== 'object') return false;
    if (!parsed.brain) return false;

    const champion = NeuralNetwork.fromJSON(parsed.brain, this.inputCount);
    if (!champion) return false;

    if (parsed.hasChampion === true || (typeof parsed.bestFitness === 'number' && parsed.bestFitness > 0)) {
      this.championBrain = champion;
    } else {
      this.championBrain = null;
    }
    const active = parsed.activeBrain ? NeuralNetwork.fromJSON(parsed.activeBrain, this.inputCount) : null;
    this.brain = active || champion;

    if (this.championBrain && this.championBrain.biases && this.championBrain.biases.length > 0) {
      const champOutB = this.championBrain.biases[this.championBrain.biases.length - 1];
      if (champOutB && champOutB.length > 0) {
        champOutB[0] = Math.max(-0.15, Math.min(0.15, champOutB[0]));
      }
    }
    if (this.brain && this.brain.biases && this.brain.biases.length > 0) {
      const brainOutB = this.brain.biases[this.brain.biases.length - 1];
      if (brainOutB && brainOutB.length > 0) {
        brainOutB[0] = Math.max(-0.15, Math.min(0.15, brainOutB[0]));
      }
    }

    if (parsed.silverBrain) {
      const silver = NeuralNetwork.fromJSON(parsed.silverBrain, this.inputCount);
      if (silver) {
        this.silverBrain = silver;
        const sOutB = this.silverBrain.biases[this.silverBrain.biases.length - 1];
        if (sOutB && sOutB.length > 0) sOutB[0] = Math.max(-0.15, Math.min(0.15, sOutB[0]));
      }
    } else {
      this.silverBrain = null;
    }
    this.silverFitness = typeof parsed.silverFitness === 'number' ? parsed.silverFitness : 0;
    this.silverSource = parsed.silverSource || (this.silverBrain ? 'champion' : null);
    this.bestFitness = typeof parsed.bestFitness === 'number' ? parsed.bestFitness : 0;
    this.allTimePeakFitness = typeof parsed.allTimePeakFitness === 'number'
      ? Math.max(parsed.allTimePeakFitness, this.bestFitness)
      : Math.max(this.allTimePeakFitness || 0, this.bestFitness);
    this.generation = typeof parsed.generation === 'number' ? parsed.generation : 1;
    this.currentEpisode = typeof parsed.currentEpisode === 'number' ? parsed.currentEpisode : 1;
    this.episodeFitnesses = Array.isArray(parsed.episodeFitnesses) ? [...parsed.episodeFitnesses] : [];
    this.candidatePending = Boolean(parsed.candidatePending);
    this.candidateSource = parsed.candidateSource || null;
    this.kills = typeof parsed.kills === 'number' ? parsed.kills : 0;
    this.teacherSamples = typeof parsed.teacherSamples === 'number' ? parsed.teacherSamples : 0;
    this.lifeTeacherSamples = 0;
    this.allTimePeakLength = typeof parsed.allTimePeakLength === 'number' ? parsed.allTimePeakLength : (this.allTimePeakLength || 0);
    this.stagnantGens = typeof parsed.stagnantGens === 'number' ? parsed.stagnantGens : 0;
    this.lastLoadedTimestamp = parsed.timestamp || Date.now();
    if (parsed.telemetry) {
      this.flightRecorder.loadTelemetry(parsed.telemetry);
    }
    return true;
  }

  _loadChampionWeights() {
    let saved = null;
    try {
      if (typeof localStorage !== 'undefined') {
        saved = localStorage.getItem(STORAGE_KEY);
      }
    } catch (_) {}

    if (typeof GM_getValue === 'function') {
      try {
        const gmSaved = GM_getValue(STORAGE_KEY, null);
        if (gmSaved) {
          if (!saved) {
            saved = gmSaved;
          } else {
            const gmParsed = JSON.parse(gmSaved);
            const localParsed = JSON.parse(saved);
            if (
              ((gmParsed.bestFitness || 0) > (localParsed.bestFitness || 0)) ||
              ((gmParsed.generation || 1) > (localParsed.generation || 1)) ||
              ((gmParsed.timestamp || 0) > (localParsed.timestamp || 0))
            ) {
              saved = gmSaved;
            }
          }
        }
      } catch (_) {}
    }

    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (this.applySavedState(parsed)) {
          return;
        }
      } catch (_) {}
    }

    this._seedBaselineWeights();
  }

  _seedBaselineWeights() {
    const w0 = this.brain.weights[0];
    const b0 = this.brain.biases[0];
    const w1 = this.brain.weights[1];
    const b1 = this.brain.biases[1];
    const w2 = this.brain.weights[2];
    const b2 = this.brain.biases[2];

    w0.fill(0); b0.fill(0);
    w1.fill(0); b1.fill(0);
    w2.fill(0); b2.fill(0);

    for (let r = 0; r < this.rayCount; r++) {
      const relAngle = ((r / this.rayCount) * Math.PI * 2) - Math.PI;
      const lateral = Math.sin(relAngle);
      if (r === 8) {
        w0[8 * 48 + 8] = 2.5;
      } else {
        w0[r * 48 + r] = -lateral * 3.4;
      }
    }

    for (let f = 0; f < this.foodSectorCount; f++) {
      const sectorCenter = ((f + 0.5) / this.foodSectorCount) * Math.PI * 2;
      let relAngle = sectorCenter;
      if (relAngle > Math.PI) relAngle -= Math.PI * 2;
      const lateral = Math.sin(relAngle);
      w0[(16 + f) * 48 + (16 + f)] = lateral * 2.8;
    }

    w0[25 * 48 + 25] = 3.2;
    w0[28 * 48 + 28] = 0.65;
    w0[29 * 48 + 29] = -1.8;
    w0[30 * 48 + 30] = 1.8;
    w0[31 * 48 + 31] = 1.5;

    for (let i = 0; i < 16; i++) {
      w1[i * 24 + i] = 1.2;
    }
    for (let f = 0; f < 8; f++) {
      w1[(16 + f) * 24 + (16 + (f % 4))] = 1.0;
    }
    w1[25 * 24 + 20] = 1.2;
    w1[28 * 24 + 21] = 1.0;
    w1[29 * 24 + 22] = 1.0;
    w1[30 * 24 + 22] = 1.0;
    w1[31 * 24 + 23] = 1.2;

    for (let i = 0; i < 16; i++) {
      if (i !== 8) w2[i * 2 + 0] = 1.0;
    }
    for (let i = 16; i < 20; i++) {
      w2[i * 2 + 0] = 0.8;
    }
    w2[20 * 2 + 0] = 0.9;
    w2[21 * 2 + 0] = 0.5;
    w2[22 * 2 + 0] = 0.9;

    w2[8 * 2 + 1] = -3.5;
    w2[16 * 2 + 1] = 1.3;
    w2[19 * 2 + 1] = 1.3;
    w2[23 * 2 + 1] = -2.2;
    b2[1] = -0.95;
  }

  resetBrain() {
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch (_) {}
    if (typeof window !== 'undefined' && typeof window.__slitherMLBridgeClear === 'function') {
      window.__slitherMLBridgeClear();
    }
    if (typeof GM_setValue === 'function') {
      try {
        GM_setValue(STORAGE_KEY, '');
      } catch (_) {}
    }
    this.brain = new NeuralNetwork(this.inputCount, [48, 24], 2);
    this.championBrain = null;
    this.silverBrain = null;
    this.silverFitness = 0;
    this.silverSource = null;
    this.generation = 1;
    this.currentEpisode = 1;
    this.episodeFitnesses = [];
    this.bestFitness = 0;
    this.fitness = 0;
    this.survivalTime = 0;
    this.peakLength = 0;
    this.allTimePeakLength = 0;
    this.lastLoadedTimestamp = 0;
    this.previousScore = 0;
    this.teacherSamples = 0;
    this.lifeTeacherSamples = 0;
    this.touchedByHuman = false;
    this.candidatePending = false;
    this.candidateSource = null;
    this.abandonCurrentEpisode = false;
    this.boostDuration = 0;
    this.boostCooldown = 0;
    this.boostFuel = 1;
    this.foodTargetError = null;
    this.enemyTracker.clear();
    this.prevTurnEma = 0;
    this.leftThreatEma = 0;
    this.rightThreatEma = 0;
    this.kills = 0;
    this.stagnantGens = 0;
    this.allTimePeakFitness = 0;
    this.coilSign = 0;
    this.constrictTarget = null;
    this.lastKillEpicenter = null;
    this.vacuumActiveUntil = 0;
    this.evasionSign = 0;
    this.evasionLatchTimer = 0;
    this.escapeCorridorAngle = null;
    this.corridorTurnSign = 0;
    this.flightRecorder = new FlightRecorder();
    this._seedBaselineWeights();
  }

  static getApexPreset() {
    if (SlitherMLAgent._cachedApexPreset) return SlitherMLAgent._cachedApexPreset;
    const b64 = 'DvERvfg2Jj7OMJ883DOMvVBUET1WmhI92mkcPRmeO71s2oM9Kh0Avvh+i75HfbW8ql+Avvf6Bb2n7xO9N1QWvkFElbwEXnA8qvwqPApxJrxWj0o9HPqmPksEoLzU+JM7lqyxu0OloLuJjOO6o9wtPkoDLrymHFo+1TLJPCDy6j3yxhi6/cBgO6SHYrvcCc07X7HJPQOAqruAVzA7CZzROKoV1DlEK3c65MoBO975mbt9H5A63e2KODRR8DswYzi4qrMxvR0Bpj/FK+K9yYPqO3ttjTw5xYQ8Rf+XPAMsiryQaTQ9zx6KvePpBb1GbvG8/f3ZvtMKQb3tqSe9Mo0AvfhkCbyUjC88AgCIPOOSh77Luzu9bSuGvFi1hrx/ubW8sZYRPGmt5zsspmY97Fjlu7sYurnMvde+fXFKvDnNWD3T3b45PMq4uqId9jpjDRU7OP0tPB9NxDldAuC7Uieguecc/z3EF7A5dBEVu6EPrbuww8e6ERUvOk6niL3gCbK4ukxJPvs93zuy3BlAlIcRPArvJjzi/oM8A7EqPB6ksjq76RU9IxVCvtaxQ73rvAy9BWpEvbHnJL2l4As+8xrOvUi8kLzzBUs8p+Hvu63I5z2lM3g+vVFbPVs4kj7muqe8t5Kzu8NhY7zfwA+7iaRvPiORX7zOdU89v+1iu/OUHT3ldvC5abiIOv/yD7qWESE8yikoO/fo4Lv8tKw5Yc1auUpaFTus0PO94wKxPtV887vJIU251NKat/hnCD2jPo09nnEpvZTuyjv3MTU7USxSQN5nszzPsSA8ppwxPMvbgDxq74g8uQHWvmSZA71r3lK9wykuvgnZ+Lzukhu9dwxWPU0nNb779UQ8Ul/fu0IRZz6Y3P68MJ2BvUEEXryQ4ya8deDUO3uMNjzDwgg7hWwTPvD+vT6+mzG852wIPrT07jwReC8+5YprO3qA9zr9jQY7lzJRPA0HV7nRFbm7uCgnutcuxr1T/BG9e3kCPQLsRDsRO0S6guyuOXQh5zqGCzk4gwC1vFVoIT4Mvb660ar0uKWPWUCVE/A75bBuOzzOC74XZQ09q/r3vFmRhLuBFd28h53XvJwhhL0aJCC9Gc0BvUG7QbwQEgM8pXUfu4AEsjveSXo+++Bxvvbp8T2HCMo+ti6hO01nTb2ML4I6i8iRuwPtuTsDs0C8AAXSPbFGqzyO3zY+eeRAOsC3HzvtY9w9ZRh8PkTS6L1fGmy7fboKvuGScDvTe2W6pwghOS/pW757EAq5DCWVOPhtFzoKFSM3WvzrvLEYITwerWg74EJoO/so8zvI4UhAIeoOOoqEND5PceI8lWipvJOooLyNvNe8nAu+vCMS6Lzgtge9HmW+vGfJE75ke/U7B2eaOyS1mz1XSNa8NEvXu/cYZrvgHOi8A2OLuqnheryJxxe7O7/yPV9h1Ls/WR+8sLDoOSL3GDzcfzi6ErFBO+Oa3LmBXx87G3i3uo5Uerr7tCC6FFdBOJAxQzur2zk9aUodOVsRLb0h7Zk5g0F7t1vZYbrSaFW+dj3wvCKb8DvTFty9pWcpO6Y7cDv+xlI6zc8ZQK9t0b0Fp/Q7KrLMvTx0TLyZF+29IuRbvDwJsbxT/+28gb3IvO0QyrvdHP29KnS8OxTdmT3BzYk72OD6u8oLF7sFk6o8K/ubPA2/wbxxxJa6dESBORr4Db5ohRq8r1/NuhWEnbvrmdm4UwREOjXV+7hXcHQ7rrNYO4rl9LX++BM+6YImuU0ptjot9Ds6BFRQvmXsVrjanpM6NzAaOjmOQDmWrT84PgLGvif/lTsy4G07mx3JOq1HUzuqA7s56dw/uckDpj8AXaw71eNOvByqPrzUOzW8KSklvEG0j7zZRba8qfGBvIV5vruN7oM7K1bQO1HfZT7v6Ge8zMu2u0irnbunY8y7eMFFO6XJu7xQJ0W49QxWunOFGLziNhy8dXGiPRPWi7sIfnG9G3XhvMj/HzqZRN86ajS3vPM6ez1pYDi6R7UattxE+jqhg7M7pMukujN3kboJVDY9LnYqPQBo3T3QywQ+fuE2vHzd2ztufd07hz7OO9TkxTsQaM09EubaO6P76jvwtDlA5coruzYQn7sQ0IK7xYgbvpAAtrvl93y72MKqu8rFYzsncW++Y1wGO87bnztbRBG7+6eHu950GLuglDc+73Bdu3Bhtrt8lJe9aZ8yO7OTCrxyFdC54Yd1O0MK4rzJuKK5vIAjOaSgCbtMxvU5x7erO/JHoT56ARE7FiFKvQna0bnafW46hP3HOuk1GLrrSOM6pBKgOjFNjT7yOAE5lP68PM4SDz49UZo8fcv3PDlvsTyxr5I8x8PYPHq2+DxpHzU8y7+lv3PKw73t6o67iONGOR1D6rrZXiu7jynKOVkGjzxO+f29tvwtPhV9Hb5S+ck7nzH2vuWjv7oWkRQ8C1qzOxXJ1zw+kfe8n33IugBvkzwcYJU7irWGPIWRh74EOAE64HPFOT98v72lBWg6ZYzgO9pptzn409O5kz+XuXzuLrk7Fey90AX6unPahrsvdtM6GIRIvtLYj755mjI4RKUmvI+BBDy7RtY740ZxPIjAPDwbgDQ8TJxnPpVcpb00Nau8BBxRvHDSGcDCze89Z7/AuZxFUrvdAw+8zCPOvckW/js7Vgg8TPQ6u+xDazyaF9q68wimux8O4rpWY6w7L9aKu7hrjzscUOe6CNqMO55HmrvLDQY7vJYvPISB+7z7Ytc5oQduu3vOzj2H0nO+0nGoO+ZzYLsrJKU7SGvPPFXIXz75iOi6UHPxPXV1A7tOWCM7f7EmO9LzJL6+tEO+ccMvvNMCeDwwRzM81MPtPZ7XFzxmDCA8Aq+WPCPE1Tw5CzO953p6vII+Fbp/BUnABHTEurGRgT5NCyO7RGQyu1+nQDw6LQo8qsgdOuh1tTwlnzu7Dry8vSRcV7t2uvO8XZ0GukfJhD7lP445NEQcOw48jT7YWp06+OGAPAHwGL2X4Bg6tgLUuhUXjbp9bXA7wR2DPM7BLj13pyo7sbPaueqtuLqaf8c6ApUPO2hbA7tFSIS+khRZO8GQOTtmfyM5Llm3vEKuxD6/IhM8UtITPA1OlDzfBJA8MYK0PcUoCj3z/+S8o97KvWz6Jb4AzpQ7AJVZwHDOzrqsVJy7oVIzvO/7Oju895I8DSOEOwbqXT7TwJO8YxSGvbeQo7tV6Ik9fwYru4RHgTzGbZa6vhVWPjghOD3tWeE7pvHRPJPnRjvXXJu5UkaLu3rqQbobO4o7nvR5PaZHTbvn5Qk+AOd3uQLHJznIKjy7T86EOw6l1Dpc+ls6ttskO+hFqzqZt8w8Jr6GvZ9voD4B2p88tUGJPMK8QDyWcIs87TFePo9I/zztTQw7yJS3vCFm1znv5Ri7nLIcvOcIScDjyEm6FOJCvDycajul4Ks+wwyeup/O8TuuABe+t2j3u+7EGbxeOQC8y+UfvGF/7Tty6r69WO2yO5sd2LsVWl48mhDZPHW3SLpGEtM+5hPNObpQZD5pUT+9Pj7nO+2pCLyNSfW93ZeQPhHut7tz5is5vZjVO8h4ozuBzKQ+0TiNvdk7mL6aCxg5tP7avPwNUz2gzPk8ggPAvYYgxb1vJNo8rsAAvoM7Nj0Xrog59BOZvNXeHLsmgcI6US03vGbzF7ws0BnAeAU9vIf9AzsQibe9lxTjOrPnB75Mn5i8e8jvu2uoJbxyitG70fbQu3QHwT33/4+7BOj9Ot9Q1DtyFZg8Xn4UPZ+RrD3sAYa+BFNzOlVckT0au2U7vKGRO6Y/pD3yaCE7UH2MuIF/kbrNe5k9q0y5PapqWzsP3wM7kyOyOilmJD4ETa44afGfvCFXCL4qtj89wbcnPf/OLD0RZi89YgFUPaphA75AMAc9CELaOWv7lD23H0c8K3eOu4hQOj5rnvm8Jealv+dygDsC5kI+fWusOiDiLzwNnJu8mQoFvP5VT7yft5S8EdV3PpZuqzxwNZa7TbT6vMriGD7PFvY8vuxCPa1uhr7MGMC6ejxCPOpjmLtT8kY8E0NSu0mJ6bsgHIU7pMUKORMADrvcHg88X/KAO3Q8BDvOdBc7W15TOfIQubvShkU45cc4vXZLdj3Jswg92yghPT0cQD07Y1C+nsISPQ8ZzTwFQqU95IxnvSVkBrw9bxU8KOCGvlbESr3z7GC9gYAjPlbchD8XgQs+uos6vvSOy7wEh4u9DK/nvXbpp7yMQus9WSaeu4LYnbzZ2xW7INhRukTST7zKPbS+3/YlPU07qj12Lv661YM5PnJjzL6CFq87S9eEvBsJqbvuxs06fM2eOuEKPT154Tw+aDcQO58otbu4/ye6jYnMvhg8s7t2wjS5sAwhPk48Qj27t9I8zEz0PNbcFD1KCPE89GITPdN82DxDGhe+37lDvTnfLb097i29PZIfvfcHNL1Aw2O99q2SvV0boL6quyVA/Q4HPAolQLwrb029yMfauzOOsLyZ2QG94iviu0ply7zSO5i7mWnYvQormbzu8Cu92k7rPF/b4DyMGie7MWbZOwfScL41Hcc7DM6lOnmVs7sPgVo72JFnOhoSwDoFTIE7CO8SPC8wgbuXAuQ5dBArv9f6K7t6xgO4J889vx5Zjj1Dy+08b6kCPZmUDj08Txc98HtTvS8CXj1CeUw+jbpUvbx2FL3/Kc28GbPrvCUViz3xb2q9B8OTPgH9pzvyHbw8khElQP5uyDuTSiK9u0CovMeklbzZ2ni8SOZ8POiDHj2LgB87bKklvkEbsDytkSu9SciJvkyKg768/4i5C/KAvamuLDuWvkM+mWySvmu6GzsSeOK7g/8NOVzloz6V/4C8/io6PPGh+To6DB26bz01O/ApDT0ohFW4SxCxvcXX4z0TUyE+0DwJPYohFz2ugeI8UgC4PI84oL3TjTm+I9gnvgb9W71TR2K9Uwf9vYI6Tr1S4nC9gya8PNNVX779K6s8zDcMO790hT+0fYC9D47YvNELubysBCm9wgi0vPopZb0Wpy+8S6divZbqXj0ZWx29RxcCPXs2t71ANk+734+HO9DvBLwBAx88XbE8vl92Frzs8R08vJtzOmFSWLtTHZk6FouKPR1hojlKevw6wAjbvZZ2pb22eNc44FCSPEuueT28FzQ9a8J4PZMRbD5ftlE9C0CDPeepyr31AAK+K7invN1qZrziziK9BM6jvMCQ9ryw5BO96AgevSO6z76u9ug8SlZPPE09hTwEXF2/xJCAvln0Z7z+k0O8NdfIvZ00ez1SRXg69oeAu3Cg7DxL2IC8cCBIPfQi/jwiG5+6BssyPBDiwDnF/Ja+UZFGPPm0Kj73kkq74BQ9vh1rlr7ygKG86twBOvn9jrs/Yfs8spMuO+yuEDsiSsa9CFS/O/fXiz1xFTs9mGxUPRK1RT2IN2s9PA/2PVk4U74890w8hu0UvSz5Mz33eaa8Zr3avI3z5L1jXju9yfw7vYMOFDumPuU8EUEkPB9OHjxp9q89DYolwLd/ibyPn128NrjKOzuiUT2buug6AzSXu38IyDzaz7689m0+PW6LDD23x5s+ftJovt4vZjpF4BO+sntFPCAkurqftNQ9Xn37OeR4uDsiOr07GUcbO0F/bbuTgZW63I6qvZLVqT1nrCU+KelOPXRkdj1p0SI9lQRePjJKXz0oKh6+lLNxPUytdD0TY0o8ACwzvYyk6LwzHLi8W+8HvYXRGb2RDPI9GT6AvWSzxjrKsus8s7lHPLQe2j1jpCC9lR4Cv5iXJcA59Va85+fBO9KCOD1yo5E+kMLHPbg1vTxQi7C8yEM4PYJ7dz4Mr6O6FT6pO9ttGTvuTwI710ZAPIcSL7qE3Gy7tqJOOuDHmb3TwxM+CIbDO6DAcLvLUMy3fzbFvuJ9yL3fHxq33tNkOid5mb5N4jI9lAQyPRgJXj1xD149ACmPvAKxSj03+QA9d2wpvSEtrL6j5tO8Tp4DvZwvNL20vWG9gjuOva9L3bsSOGC+jE4VPGjjPT4lnlm9WSKzvZ4no7zzjIq/KDqpug0Nqz430wK9iGy0ujFkRzyQ6QC9XaoXPoPDTT19Lt26RSg9PMbvRz5w0qA9NJgXOmB0UbvGBnI51wiDOn01YTtx9607igP3O29Awbrc18+3O4PSPRctqLpKnlO4DNw1O5ZAHzwnVZY7B62/O7XABzzs8QA8N+bmPZtMpTvklOI8Hj/Ou76XiLu1Mza7Dm2pu+vAu7v3YNK7bGAlPp7Zp7slt6I7v1FxPlygWLtUrDe85xGjPjDMbrt4dui7dU/bOEraODtgZiY5gb+OuiIxWr6MdFI9SVvmO8bycrv3FrW5iLu/Og+GBD6cn/c5apHNuyQj8Lm7kNO5sjalPVU9iDrAb+o4xKq9PeawJj4GCL+5rp8zvioAg7peH5q9AAAAAG5+0j0AAAAAAAAAAAAAAAA2l829AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv7Y8vDLoFz0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgokPAAAAAAAAAAAAAAAAM3MTEAAAAAAQrIXvm2zCb008T28LWEwvgAAAAAAAAAAAAAAAAAAAAAIKOy+AAAAAAAAAAAAAAAAroM8vgAAAAAAAAAAAAAAAAAAAACGzTK+AAAAAIljkD4AAAAAUwVgu/giJDzUysQ7inDcO9Hk+julfec7G5nzOz+u3zuj2CI7ljTcuxKi1b3935C7nky9u5Vwxz66MmA++D1kvMqGNT0jAYo7k77rOiHKQboejgK8RNFqu7L+Srv+H34+eIe9ufd8tD1R7O65IGcGuZPLrzk3LqG7vRDdOzcw0zus2IG5P4WOOh4Rub3nDAm+cxvTPfKNWj7gNsE4J2fWOPphUb7VlSA6mwGQOttoSr2SExC97RrMOQzbk75i0MW2CZe1vJEeJLsg3HC6TQ45OtWkyr0hxBO8ZFKSu2hsHjsLkB6+47z2N5ULsjvIKos7nPuRvPDjBjoqjUw6GFjLvc/ctTybzVQ2QRIAuWxNtb6VHDI8duCkuIJj+j3C9ZI8nPnjvVNeo7m9hGu6lRyCO1/SELwt2ZA+/bSNuzQ/iL30BjM5zUR+us5TgLxH8Rq+XN9XPthpALmAVKK9yWeHughIaj0tY0c7+j4XPMjsEzu1afU6MyHIO0sNlzuBPbk5r9kfvD+wzT3ATbG7bj24uwOFYbtjIPu6b+GAuyWGOj7Uteo6J1CMu4L2wrrWrFe6ZDHQPD0ynz5LKqu7WIP/uzXzp7uB77A7jDACuwU6Fbsw33U+RPWGOfqvs7kwFGu7R9hwu8JEPLxcfZW6hZrOOiTuJD8l+LS5sACeujmhzbh9jbq3DJU9vsyInD2HoTU7RyP4vSXbBz6hWSU7ED4RvvluSrvSGYO7yjEVvh9vlj5/2b05lictPsYuwrrHueA2mxyBPRDHBr4UKFE9tFCBOxFwjD0sEiQ+OnO5PbsJoDvc96o9Kp69vBftTLz4W5S9zqqhvafCe7z7B0m8N9GAvM44FDvchNI8K5CPPKWsMDzn+DS81uNZvPP8YzvVc4s9oyxjO2z2jD0KzYk65iilu5Q5OD1+sMq/BvNFPWpvqT0Wkg69qGIYPLFCIrrcMJs5CDmouzA1xrpC9SY90Zy3ORbelT1HmwA+lTVWPmIe4bvCOSm95uY3ucS7EruS4AC5wi++vTsIlDycuio8uDSNPDKltjzRx5s8Z4p5PPBHJ7v3HDY9zM+kvWVIjL1WPgy+iK7mvvb9hr0sHqY7g8prvYsGIL3TJs89VH+sO145Br00loG92ulPPdH9s7wVpFu8K2xcvAHrgb0V25m7vwYjPN6KNL3ncDK9KD3mP2YFTT1SIN26xmIVu1QpobvtvAo8gklsvIe07bsIX7879KshOrVst7ogCKC7k8OoO2jaKr5DTac6huOTveo9x7uepu42TJBhvLGxZjsFeFq9gafeOsT6DjvAkbg6ukz2OcyDVDofhcC8vEr+vcgMqT1v4YK972xfvJoa8j2b89y7VDBqu9v5BjuIfYY6bq4euhTJizrSKuY+J5AlPowMaD0uFps6LBeYut19Ar43Zyy61jrFPmm+ibuXFw+725eUOp9wvz8JcJS4RLzwOQDdaLrYKro91NQ1vYN7Qro+vDU6xjJmuQc/dT4w1WE6AcK0Oq6Jrb3e5/w5h3iKOqFqizfI8Pi9+lySP6qDhbzC6h47iN3gu1g0hLwI8su8xEXEvBsNCL1REKC9oO+tvDuD1by2zs68oGChvJDYoru4mGA7ztFFPH6MJb0omnG9eBKxve7Kfr1v20O9UqIIvftWa77+fFE8/AAVvYZrlj+h7iK7+wFCuhSUxTmNX/07BY67O8pCXDvMuWy8YlRyvchMLL3D6UO9yvboPbtb8rys8ia9RGflvGABfr1xUTe8SdV6vMItzLwI5lu8sRxbvEAhiLzLwBM9FieMvamHED2gRZk/VNdEO8V1YjoiIeU7igeQO9CYz7qb7a67aL9pvfVvQ72CT0G9tx2PvZOcDL3ik/i9jPruvNhAib23EmW83fi8vC3p7jglPMY8NxKOvOL0bbwZwyk9R4hTvf5NIzyeR486E2aZP4LiID5ltIU7n4C5OlVR3jmEL2q83iFHvSUaHL1OFDm9saE2vYBvE72g+Ny8zNg3vdvNgb372Yc+iaedvFwJZL2ckM28OXuDvC+6uT0amwE9wx9JvRzinzroese7UQaxu0G6mD8IdSm7d3sFvEUYGrzIsLm84EhZvU4cMr0yuVi9QAMvvalTdT65OGm9rJMJvo/pyj0G+kW86ZfQvISTdr3lrgG9eqeEvFOA97zQzA4+KM5IvRieIr2J7xU4MSW5unGXGzpDr5g/hAUQvnGBJLzfbka7fESIvkyp/L2kPkO9C0+Pvs7xJb2n+0u9Dr0rvVatdr4gmGu84ei0vN9hWr0kFHo8SAiQvA2G17yaL9Y8ZVhLvdS6Vj6rlqk5+7JOu1nplbug2v+7DlOYPwhjfz4hoq28SxguvSK2A70U8yO9F6z7vMgKbr7giDC9IegYvZGQLTwm45m80NHbvNHcN72UpBW9G9OlvH1QI70ZSWc8GwXwPAcUmDvHdj27kXtRu+m1l74DcB+8hhtWvvs/mD+iDzS9kikevV50Jj4D/wS9E+llPdgzsbzGSey86obnvPlwmz6K2o286xK8vHtcD70n+fm82juLPUaKRL7jqSE91At2vFt1ozyXEHM8cj+8PbLBLTxqaAg8bbpLPGoGYTy48ZU/uYLPvVqHK7yTsYW82IOIPoUq3bvxIrG9t0I1vH+HDLxYYRQ8HlYWvOAc9Lk0wPm6pc41PkZoH7w9cci83LoSPUjiCr24M/e8CeMYva4KIr2sezW9TxlOvSy3UL2HGKg8NzaXP3+Zs7wzEna8IygUPsHnKrx4Mc86DQS6u8seED4TanW9yY1HvYS/y7xdSEG+OdP8vJ93Rr04G7K8M5A/vPgxnL54drq86mbkvHcHBT6BF/O8qcgbvjPnKr3vERE92MlDOt+Vlz/7IGK7nXt3u6pzlbtQ/zI8XI4EPHhJAzsMzBS9pDKuvrbCQLz6Vuy82FmBvPWysrzHzk68GOvUPFqtGb21dwC9Ua8CvWVtB720zQ29SCE3vYyyar6GoDk9TP6jvsQchbxEzZc/l8INvDJRFbxhkyo6KC55u+mOKL4X5j69eikIPrejhT7griW9ynO8vMmwqzzDd1O9A7XPvAA+WbxMIAC9kbIRvbHeL70pRJU+7kxqvYKPcr1FxoO8g2rcuT7gYrzEl627hOWJP2xXBLzdi9w73GmjO+vuOD3JGj+9dZZLvaqLXTzrHSa906THvPDp47w4WF+9nZtyOsMgVr2JXxi9a7sMvTwt8bzt7wS9/dMevkn4QL2r5cw8sgAQPEIQArylwpC8KWVoOz1KkD+qMW4+k1bQOzC0eL4FPc69bGzYvkQu4ztztFy9g1Z8vNqLd7wa4ue84xmavKTGbL0wAym9KlQevQIYGb3FOy291s04vTWWbL3ATrs8ap+6Ok7oIbyogJg+9gatuwhcZbxhm5g/TddJvQy3TDt442u9ugMsvdWMKz5wkre9UH/YvItABr1W0J+8F9ZKPth8jr23dla9tC00vWbJgr6uQUO9HTdQvRUjKz5KTAU8QPATvH5+X7yGqzi8y+OZu6SzsT1vd9w7xouYP4KaDzxi2jC9riv5vN87iz5x9A+9+GSovGOMkL5Way69I224vIMg0TxhGwE+BUaEPFABhTzBO0U8HX1vPGkmTzz77/M88RYYvQIr0Lx+o+u8efXwvN/XurxgGMU+0hoGvfw7WD8TMv27CXIVvCTfSb7i6SU+WbETPf3Tm7wk8pY+wGAuvYFVDT2pFZU8DkCSPD9rsTyU7IE8UUyGvmW4kDzWnf+7c+tGvVUlD70qbEy9rwfsvcM7O77A/Uq9tqEAvivymL1PSX4/KsFxPj4Qi71yEIM+yEV4vGFdjrzJFSU9juWwPJiRLD31Qtk92/OQPAzKozwygKu+x6adPK6qqDyhm0a+XCZJvYIN/7yxzjm90s0qvWk3Eb123U69YD+9vZljg73Wpss5+cx8P3qVbL3JI268JlMnvBbgvLytPxo9cMohvoskxDz2zAo8/sAJPLQpAzxzYQO+mjM2uVI+AL6xFDC8QdFiPf0/eb5CLlC9YxpIvT87Dr0l2zm9M0o1vS1Jij4v5pG+91DnvOBLbz/6q2M7+ADmvI6YW73KaQA+foysvFE5dL62byO9BwcwvW5sOL2d4I29li1sPP5nVb0y4Vc89vI1O1SvJD4XrHc+/NAzPhJuMbvyf8w8DpFGPAR6gT+LIEO9IIgdvRjebTsijyO9UeAnvthi+rxQqfu8jYsSva+s3zwnmEi9/bVjvSRjar3VD4K9mnuBvfJWhL0WvN8zgvnaO4lc0Lv7iDw+6mDauofcADlwnCO+Ei7CPYwxxzy01Ws/3d7UvOcYH7tnhGS9U0MRvZh7ij0mYjS9h1fqvFHMdL2hTiq9iaQ8vfr3Y72R8VK+8Aq8O0zbc7383eY7gDN1PFYIjLq9BJc7zjQNPHxoOjtxQP+9dHjTPA2T4Txr2Wq9g612P81yhDvuuzu9BsoQvu5DHb2D7mC9qzrxuzVzTb2ZG8++KXsCvdJEG70O1xi99c2wvAGBGb1Wixu6MkYtPZQBLzs+slc8nZznvBXBSzxD1to8v6n2PLBLQjxAJP68f8mWvs2AVT+45LK8aN1WvHuJjrzsUBS9pG9FvLEEjTuQcmO96vsLu1uVHbs3bCu7zi2ouz8rdrtfsn08ZsmGuj9YHz+3sZ67UuSVu0bMdrumSwK882tyOW1dvj3IJKy7PkDpu2ZzcrxwHOi7s8S/Pqnrbr58Wyk8O7bKvBMWJTzPPdg8oNzHuikyprsyjd67nSDmPWH5F7yUkma85a6kvNr+Ej58QfU97Yrnulhs5Lpytga8/OM8vBx5wLxH8QG8XT3hPTFvpLzksZc/6vT9u56ztbzQS5i9yubWO2ENZrriDcE5jJiNOrhdGTslvBI7HhU9O6gzFztQBcE+CE0UO+engDrK7Zk+o+2pOnGziDnPDKG5X3zyOpojxrmmoog6E7YsO0zURrr7rCA7/GcDO/RwbTuN8l87OwwDPCR/hDxcytc5c4KROoLzJDsbeuG9xIKDO9AzmL0avna7P4KVO6uakDuVW+s7oDLLO5GrMzsMIy0+2s88OwxNCDyzVVE8U7IUPB5BNTwBejo8Yq1wvgcGaDzHRzA+gYeIvCYGIz3AGG2+fmTtu58XAD4L3ii8zrDcvIVyF7wi3Vm7mjlPvNmQDLxH5B+8Esr+u/+xr7s8yOS72H0MvJzVoz25+oK8Vy56vGhsW7xqm2+8qdR9P/bhSb7moOq6gZJQvb7agb2JkFC9QXzCPe11hL2AapG9T46hvWcdmr3gEBi9mP5zuzc8eTxZcxY2hCMXvINQcj0927G76id7u4QQuzzxNY69lmeCvS8ttrxhW1+++x4kvaYXdD9pbY6965z7PQPFWrqRRqi7VQumuxmFtburiie89DWnuzArIryHflw8QaGhvXsxgr14l5G9B6/HPh4kVr2l22+9gZNovdNw/b3OyyC9yuQtvePJ2b2uxky92KZBvGOWcT8NyaM+GnJavQs+HjzSth48VO+6O0wEgTuhh8w7sBLEO1dWoD0N/EK+y5NivO6+6T4WWuI5KXmxvUFj3ru9yho9c/yWu/y9BLylb1478MPFObcZPLufZVs6/UTUukiVRrv5n5o/JdKUu9bqqTsJ60M7PLlQOpD4f7qiiZi9CVwPO8xgWzt2VoS8bvxPu+3oDjvtmNU5guHfvhN8lzmFlMG7wk0bu8X8Ursa5fS8XceKO0GpK77H97E7OhoUPt+Ijju6N6I6Y7uXu4xluz7UWAs7oSnbOvG0Fj1j3Ng3Bg8NO4uM/TnCgqk7ROW/uhes3T0pBra6YK8tupf4tjo0Ahg88f15OmwoHLwOImk6Ka5XOl5gFrwjYU66KP1VuszXHj28MjI8DsqaO9lXA7zU/3e+WuvmuuqhXLu8KBK7boxgPipsjrtoAgK8zoNaO2iGB7v2cBA6I8lQOYDx9zy9usk6xGlPO1p9kzyIVsK7W/QPvtA5fzxmxyC7tGoSu/niUT7RYV+92SnXvdmggj2pbyy5djRkOhzERzvasVw7ivSMO7kDlT5AGEc9NSmLveMsGT1xiUo7b7/zOokLzTqFW4++bJm/O6CZQT3+8JM7cwSjOwhbBrzGkJI710mcO1vHCzyyiS28q2e8O909Zr6cCzw5/HuMOI0nczpgu4w74g5FO0p5jzuXZh2+MksVO05uGDsqs6w71HJiOy0RoD2A6e49AdFTPZbqwTuP/aU7YSYAPSphkT7O0MI7RcuRO/O/CTwdb5s7OPfRPZwegzqyvaS8DLtvutTqe7onTIS7wl9fu9XRWLu4Pay84tfLvWYB4705BPU8HbByu0jQPzqmqwc+pDOiulpx8zv7SWK7DgEku3aT8zq3Dpe7ovOZu2vm9LvBm+486gHsOS5OB753erw5WiaePFL2yzm7O9W7792qOUiZhznwSJu6WKmjvR+OjTkYJgE6fyXwOWITyzkh0yi+PMrlOfrVBzpaOhS9cQUbOl9SVzoRSDi+1ofAOYofBTrx3sS4ZIYaOyJLprvilLO6xOeTuugX9L1k+6a6GPYiu2BeK7vBDCI88WaZOrSYEbtABLK662RvulfrH7suX8i4wAgOOnSpljk3+ZK7XWUyu45mFbllxpi7od3MPbYxmzndrs66bHARvKBNWbtnikQ6gBxRPoSaAbrcRbe8a4vZuTr0b76BWYG8CC2Du0G1d7nIbwi7HX64uvzoorogYwu+0+FKuy0gR70NzUe5z2xcPmYOILs+XAq7PXGiPUKtnLsgBJE+kIz+uxq6pz5Xjqo77xKPOUdpiT3HD6A9szuJOulTnzkBRpS7x+kuPHwTqr05hCq951l/ujOxPTtHwLY6uIHsOGiESL1AwnE7wS7lOj8txrubQHI66WSCOZ7pWLtdflS9e+0EPiZN2jwzKcg6cd3WOiFJszr5cQI81O2kOzVYbj08XVC9AERSuwg0fT5E6tE6QHHMOsYSmDt04m679ZOAvgi287s+uBM8xEu1O6GWD7zJMaM7jthRO63aSb3ER7A8ef6uO3d4wrvYlS87C3nMPZf9F7rc79i9Yf7LvtiJzruqhKU8DMCjOz1HXT120ug5hxkYvvLmz74vsAw8QeEgvnZ9tD0HQQe+1/i7uydYhbirvrm7njSMu2ZwkT2zv5w9SeDPvmMPAjsAIV+6nrB3u4UR57tw83K7RRi9u5ATbr5rTYS8kTW/uzxa/zxM6Nm9lrRjuwySrDrSSYY33F2auy17x7qAGlk+APa7u/pVs7t876e94WR2PmcX5ruwUQk7qW2BPEKaIbymny+7eE4su0E/iLv83Pm6Px/Auwmdi7vet0C98JcSPEnaSr3gpbk7Q7hvO9eLKT55bOY7od00PPmHYrwqaJ67rhMgu9gSnTzST9A6aod4vfHDKDw4I6m99u5hO2T/CzlJpee6XoevutJDprr1v726idjWug3xjbpDmnW89bedPozehrkWqQw50qk+uGk32Tw651k6oK48PosyCjs1cNe5dQjHuGGnsDrCheo9HlDQuemunjonvyg+c/1luqBB/z1Oywq5Ly1cuQcVuLnR5Lm5Sjn7uWNQz7mif2W7wArJuYon8z3ke+Y3uq3ft3nSH7lsL4C5f9+cvcroH736zMq8KguUufxVmzos0ye5W3SSubPYsbnD+6u6E11nP1EcBr+P6oM/OM+CvNN/ej/TSGK8yz96P5btNLzNlXc/tXJLvONHeD+FcTG8IHR2P8ZQnb6fnnY/RTDbu35otbtnXV7AJbpxP9eEsL2P+2c/8BX1O4i7dD+piZI9n6J3P4S/Db0x2Hg/Uo45PJ+Sdj9ZyvQ9H/NzPxnxdTx+ZS0/i/GvP9tdMj80KIw7nLo2P9tWkTsfvjY/FYWnP8H9Xj/uaBO73bTuPhBD/r3vQD0/70QZPORbdTlRAQ3AMRiZPu8Dkz2rzSs9KimQPnCZaT3Telw9apBgPVmDSD10kX09mjZGvTcpFL2k3wC9VKUrvZUCq72XfG29BGeUvS74sr7d9/w8Vn9uPfd2KLwHWho+xq8+vfqlt7yl9gq9S8NtvkOFaDxYgQK7Tj4tu8tM1zv9Mwy9WXZKPWPbwjwPAga+XebQPRLtjD1toLY7QhTqu/Cldbv8vBu+yRegOuk+hTuIA2A76ZmrO2fYkrtUARo+fGKyOqFDN7sWqJk+Puasu7eZVT3J09K98WoUPZnHKj2F2Bw9iAAsPQnPID0moJg8x2xdPtiQGr6cF9+8ue/svONJeb4mble90z8KPiJ3gb2F6vU6iF90PHK6Tr1JFxW+d9XoOxmhcTvyy2Y9kJvdO4qJeL8=';
    const raw = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const f32 = new Float32Array(bytes.buffer);
    if (f32.length !== 2810) {
      throw new Error(`[SlitherML] Corrupt Apex preset: expected 2810 Float32 elements, got ${f32.length}`);
    }

    const w0 = Array.from(f32.subarray(0, 1536));
    const w1 = Array.from(f32.subarray(1536, 2688));
    const w2 = Array.from(f32.subarray(2688, 2736));

    const b0 = Array.from(f32.subarray(2736, 2784));
    const b1 = Array.from(f32.subarray(2784, 2808));
    const b2 = Array.from(f32.subarray(2808, 2810));

    SlitherMLAgent._cachedApexPreset = {
      brain: {
        inputNodes: 32,
        hiddenLayers: [48, 24],
        outputNodes: 2,
        weights: [w0, w1, w2],
        biases: [b0, b1, b2]
      },
      hasChampion: true,
      bestFitness: 37089,
      allTimePeakFitness: 43634,
      generation: 100,
      currentEpisode: 1,
      episodeFitnesses: [],
      candidatePending: false,
      candidateSource: null,
      kills: 168,
      teacherSamples: 1079717,
      allTimePeakLength: 22783,
      stagnantGens: 0,
      provenance: 'checkpoints/slither_ml_brain_gen100.json (Century landmark; 100 generations of live play, 168 kills, 1.08M lessons, 22.8k peak length, 37.1k floor)',
      timestamp: 1793231453303
    };
    return SlitherMLAgent._cachedApexPreset;
  }

  loadApexPreset() {
    let preset;
    try {
      preset = SlitherMLAgent.getApexPreset();
    } catch (_) {
      return false;
    }
    if (preset && this.applySavedState(preset)) {
      this.lifeTeacherSamples = 0;
      if (this.championBrain) {
        this.brain = NeuralNetwork.fromJSON(this.championBrain.export(), this.inputCount);
        this.brain.mutate(0.035, 0.08);
      }
      this.enemyTracker.clear();
      this.prevTurnEma = 0;
      this.leftThreatEma = 0;
      this.rightThreatEma = 0;
      this.foodTargetError = null;
      this.boostDuration = 0;
      this.boostCooldown = 0;
      this.boostFuel = 1;
      this.abandonCurrentEpisode = false;
      this.kills = preset.kills;
      this.teacherSamples = preset.teacherSamples;
      this.flightRecorder = new FlightRecorder();
      this._saveChampionWeights();
      return true;
    }
    return false;
  }

  _pickFoodTarget(currentAng, headX = 0, headY = 0, snakes = null) {
    return pickFoodTarget({
      detectedFoods: this.detectedFoods,
      currentAng,
      headX,
      headY,
      snakes,
      lastKillEpicenter: this.lastKillEpicenter,
      vacuumActiveUntil: this.vacuumActiveUntil,
      slipstreamTarget: this.slipstreamTarget,
      enemyTracker: this.enemyTracker
    });
  }

  _extractInputs(snake, snakes, foods, preys, dt, now) {
    const pageWindow = getPageWindow();
    const arenaRadius = (typeof pageWindow.grd === 'number' && pageWindow.grd > 0)
      ? pageWindow.grd
      : (typeof window !== 'undefined' && typeof window.grd === 'number' && window.grd > 0 ? window.grd : 21600);
    const headX = snake.xx;
    const headY = snake.yy;
    const currentAng = (typeof snake.eang === 'number') ? snake.eang : (snake.ang || 0);
    const scale = snake.sc || 1.0;
    const sensorRange = this.baseSensorRange * Math.min(2.0, Math.max(1.0, scale * 0.9));
    const headRadius = 14 * scale;

    const currentScore = (typeof snake.sct === 'number' && snake.sct > 0)
      ? Math.floor(snake.sct)
      : Math.floor(15 * (scale - 1) / 0.001 || 10);
    if (currentScore > this.previousScore) {
      const growth = currentScore - this.previousScore;
      this.fitness += growth * 25.0;
      this.totalFoodEaten += growth;
      this.previousScore = currentScore;
    } else if (currentScore < this.previousScore) {
      const loss = this.previousScore - currentScore;
      this.fitness = Math.max(0, this.fitness - loss * 6.0);
      this.previousScore = currentScore;
    }
    if (currentScore > this.peakLength) {
      this.peakLength = currentScore;
      if (this.peakLength > this.allTimePeakLength) {
        this.allTimePeakLength = this.peakLength;
      }
    }

    const combat = extractCombatGeometry(
      snake,
      snakes,
      dt,
      now,
      this.enemyTracker,
      this.minBoostScore,
      this.boostFuel,
      currentScore
    );

    for (const [id, record] of this.enemyTracker.entries()) {
      if (!combat.activeIds.has(id)) {
        if (record.wasClose && (now - record.timestamp < 1000)) {
          this.kills++;
          const victimScale = record.enemyScale || 1.0;
          const myScale = scale || 1.0;
          const killRatio = victimScale / Math.max(0.1, myScale);
          const combatBounty = Math.round(250 + Math.min(350, Math.max(0, (killRatio - 0.75) * 450)));
          this.fitness += combatBounty;
          this.lastKillEpicenter = { x: record.xx, y: record.yy, timestamp: now };
          this.vacuumActiveUntil = now + 6500;
          this.flightRecorder.logIncident('KILL_CONFIRMED', {
            t: this.survivalTime,
            x: record.xx,
            y: record.yy,
            info: `Kill #${this.kills} (+${combatBounty} Bounty, victim scale ${victimScale.toFixed(2)})`
          });
        }
        this.enemyTracker.delete(id);
      } else if (now - record.timestamp > 2000) {
        this.enemyTracker.delete(id);
      }
    }

    if (headX > arenaRadius * 1.05 || headY > arenaRadius * 1.05) {
      this.isCenteredWorld = false;
    }
    const isOffsetCoords = !this.isCenteredWorld;
    const cx = isOffsetCoords ? arenaRadius : 0;
    const cy = isOffsetCoords ? arenaRadius : 0;
    const relX = headX - cx;
    const relY = headY - cy;

    const rays = castProximityRays({
      snake,
      snakes,
      arenaRadius,
      relX,
      relY,
      headX,
      headY,
      currentAng,
      scale,
      sensorRange,
      headRadius,
      rayCount: this.rayCount,
      enemyTracker: this.enemyTracker
    });
    this.activeRays = rays.activeRays;

    const corridors = findEscapeCorridors(
      rays.activeRays,
      headRadius,
      currentAng,
      rays.encirclingThreatRays,
      this.rayCount
    );

    const radar = extractFoodRadar({
      headX,
      headY,
      currentAng,
      foods,
      preys,
      sensorRange,
      foodSectorCount: this.foodSectorCount,
      mySpeed: snake.sp || 5.9
    });
    this.foodSectors = radar.foodSectors;
    this.detectedFoods = radar.detectedFoods;

    const inputs = new Float32Array(this.inputCount);
    for (let r = 0; r < 16; r++) inputs[r] = rays.inputs[r];
    for (let f = 0; f < 8; f++) inputs[16 + f] = radar.foodInputs[f];

    const distFromCenter = Math.hypot(relX, relY);
    const normWallDist = distFromCenter / arenaRadius;
    const wallPressure = Math.max(0, Math.min(1, (normWallDist - 0.55) / 0.35));
    const angleToCenter = distFromCenter > 1 ? Math.atan2(-relY, -relX) : currentAng;
    let diffCenterAngle = angleToCenter - currentAng;
    while (diffCenterAngle < -Math.PI) diffCenterAngle += Math.PI * 2;
    while (diffCenterAngle > Math.PI) diffCenterAngle -= Math.PI * 2;

    inputs[24] = normWallDist;
    inputs[25] = (diffCenterAngle / Math.PI) * wallPressure;
    inputs[26] = Math.min(1.0, scale / 8.0);
    inputs[27] = snake.sp > 7.0 ? 1.0 : 0.0;

    let maxLeftThreat = 0;
    for (let r = 9; r <= 15; r++) {
      if (inputs[r] > maxLeftThreat) maxLeftThreat = inputs[r];
    }
    let maxRightThreat = 0;
    for (let r = 1; r <= 7; r++) {
      if (inputs[r] > maxRightThreat) maxRightThreat = inputs[r];
    }

    this.leftThreatEma = this.leftThreatEma * 0.85 + maxLeftThreat * 0.15;
    this.rightThreatEma = this.rightThreatEma * 0.85 + maxRightThreat * 0.15;

    inputs[28] = Math.max(-1.0, Math.min(1.0, this.prevTurnEma));
    inputs[29] = Math.max(0.0, Math.min(1.0, this.leftThreatEma));
    inputs[30] = Math.max(0.0, Math.min(1.0, this.rightThreatEma));
    inputs[31] = Math.min(1.0, combat.nearbyEnemyCount / 6);

    return {
      inputs,
      minObstacleDist: rays.minObstacleDist,
      frontObstacleDist: rays.frontObstacleDist,
      frontThreatScale: rays.frontThreatScale,
      nearestThreatScale: combat.nearestThreatScale,
      outerPredatorNearby: combat.outerPredatorNearby,
      leftClearance: rays.leftClearance,
      rightClearance: rays.rightClearance,
      forwardObstacleDanger: rays.forwardObstacleDanger,
      normWallDist,
      diffCenterAngle,
      headRadius,
      currentAng,
      currentScore,
      cutOffThreat: combat.cutOffThreat,
      attackCandidate: combat.attackCandidate,
      constrictCandidate: combat.constrictCandidate,
      huntCandidate: combat.huntCandidate,
      closestObstacleType: rays.closestObstacleType,
      frontObstacleType: rays.frontObstacleType,
      closestObstacleAngle: rays.closestObstacleAngle,
      frontObstacleAngle: rays.frontObstacleAngle,
      minLethalDist: rays.minLethalDist,
      closestLethalDist: rays.minLethalDist,
      closestLethalType: rays.closestLethalType,
      closestLethalScale: rays.closestLethalScale,
      closestLethalSpeed: rays.closestLethalSpeed,
      closestLethalAngle: rays.closestLethalAngle,
      frontLethalDist: rays.frontLethalDist,
      frontLethalType: rays.frontLethalType,
      frontLethalScale: rays.frontLethalScale,
      frontLethalSpeed: rays.frontLethalSpeed,
      frontLethalAngle: rays.frontLethalAngle,
      selfShieldSide: rays.selfShieldSide,
      frontThreatSpeed: rays.frontThreatSpeed,
      nearestThreatSpeed: combat.nearestThreatSpeed,
      encirclementEscape: corridors.encirclementEscape,
      slipstreamTarget: combat.slipstreamTarget,
      corpsePelletCount: radar.corpsePelletCount,
      maxCorridorDist: corridors.maxCorridorDist,
      bestCorridorAngle: corridors.bestCorridorAngle,
      encirclingThreatRays: rays.encirclingThreatRays
    };
  }

  evaluate(snake, snakes, foods, preys) {
    const now = performance.now();
    const dt = Math.min(0.1, Math.max(0.001, (now - this.lastDecisionTime) / 1000));
    this.lastDecisionTime = now;

    if (!snake || snake.dead) {
      return { targetAngle: 0, isBoosting: false, rawSteer: 0, rawBoost: 0 };
    }

    const survivalBonusRate = 4.0 + Math.min(6.0, (this.stagnantGens || 0) * 0.5);
    this.survivalTime += dt;
    this.fitness += dt * survivalBonusRate;

    const scale = snake.sc || 1.0;
    const state = this._extractInputs(snake, snakes, foods, preys, dt, now);

    const foodTarget = this._pickFoodTarget(state.currentAng, snake.xx, snake.yy, snakes);
    const richCorpseNearby = Boolean(foodTarget && foodTarget.isCorpse && foodTarget.clusterCount >= 3 && foodTarget.dist < 850);

    this.closestObstacleType = state.closestObstacleType;
    this.frontObstacleType = state.frontObstacleType;
    this.selfShieldSide = state.selfShieldSide;
    this.encirclementEscape = state.encirclementEscape;
    this.slipstreamTarget = state.slipstreamTarget;

    const outputs = this.brain.predict(state.inputs);
    let turnDelta = outputs[0];
    const boostProb = outputs[1];

    const threatScale = Number.isFinite(state.frontThreatScale) ? state.frontThreatScale : (state.nearestThreatScale || scale);
    const threatRadius = 14 * threatScale;
    const massRatio = threatScale / Math.max(0.01, scale);
    this.threatScale = threatScale;
    this.massRatio = massRatio;

    const threatSpeed = (typeof state.frontThreatSpeed === 'number' && state.frontThreatSpeed > 0)
      ? state.frontThreatSpeed
      : ((typeof state.nearestThreatSpeed === 'number' && state.nearestThreatSpeed > 0) ? state.nearestThreatSpeed : 5.9);

    const currentSpeed = (typeof snake.sp === 'number' && snake.sp > 0) ? snake.sp : 5.9;
    const mySpeedBuffer = Math.max(0, (currentSpeed - 5.5) * 15);
    const enemySpeedBuffer = Math.max(0, (threatSpeed - 5.5) * 18);
    const behemothMult = massRatio > 1.8 ? 3.0 : (massRatio > 1.4 ? 2.5 : 2.25);
    const dangerDistance = (state.headRadius + threatRadius) * behemothMult + mySpeedBuffer + enemySpeedBuffer;
    const dangerSource = Number.isFinite(state.frontObstacleDist) ? state.frontObstacleDist : state.minObstacleDist;
    let dangerUrgency = Math.max(0, Math.min(1, (dangerDistance - dangerSource) / ((state.headRadius + threatRadius) * 1.5)));

    if (massRatio > 1.4 && dangerUrgency > 0.04) {
      const behemothEscalation = massRatio > 1.8 ? 1.6 : 1.35;
      dangerUrgency = Math.min(1.0, dangerUrgency * behemothEscalation);
    }

    if (state.frontObstacleType === 'self') {
      dangerUrgency *= 0.25;
    } else if (state.frontObstacleType === 'head' && dangerUrgency > 0.05) {
      dangerUrgency = Math.min(1.0, dangerUrgency * 1.35);
    }

    this.evasionLatchTimer = Math.max(0, (this.evasionLatchTimer || 0) - dt);
    const rawSign = state.leftClearance >= state.rightClearance ? 1 : -1;
    const clearanceDiff = Math.abs(state.leftClearance - state.rightClearance);
    const totalClearance = Math.max(1, state.leftClearance + state.rightClearance);
    const isSurrounded = state.inputs[31] >= 0.33;
    const flipThreshold = isSurrounded ? 0.35 : 0.25;
    const decisiveAdvantage = (clearanceDiff / totalClearance) > flipThreshold;

    let reflexSign = rawSign;
    if (dangerUrgency > 0.08 || state.cutOffThreat || state.normWallDist > 0.88) {
      const latchDuration = isSurrounded ? 0.65 : 0.35;
      if (!this.evasionSign || this.evasionLatchTimer <= 0) {
        this.evasionSign = rawSign;
        this.evasionLatchTimer = latchDuration;
      } else if (rawSign !== this.evasionSign && decisiveAdvantage && this.evasionLatchTimer < (latchDuration * 0.45)) {
        this.evasionSign = rawSign;
        this.evasionLatchTimer = latchDuration;
      }
      reflexSign = this.evasionSign;
    } else {
      if (this.evasionLatchTimer <= 0) {
        this.evasionSign = 0;
        reflexSign = rawSign;
      } else {
        reflexSign = this.evasionSign || rawSign;
      }
    }

    const wallDanger = state.normWallDist > 0.91;

    let stance = arbitrateStance({
      state,
      currentScore: state.currentScore,
      scale,
      dangerUrgency,
      massRatio,
      normWallDist: state.normWallDist,
      wallDanger,
      cutOffThreat: state.cutOffThreat,
      attackCandidate: state.attackCandidate,
      constrictCandidate: state.constrictCandidate,
      huntCandidate: state.huntCandidate,
      outerPredatorNearby: state.outerPredatorNearby,
      richCorpseNearby,
      lastKillEpicenter: this.lastKillEpicenter,
      vacuumActiveUntil: this.vacuumActiveUntil,
      now,
      inputs: state.inputs,
      headRadius: state.headRadius,
      maxCorridorDist: state.maxCorridorDist
    });

    this.combatStance = stance;
    this.attackTarget = (stance === 'ATTACK') ? state.attackCandidate : null;
    this.cutOffThreat = (stance === 'DEFEND') ? state.cutOffThreat : null;
    this.constrictTarget = (stance === 'CONSTRICT') ? state.constrictCandidate : null;
    this.constrictDuration = (stance === 'CONSTRICT') ? ((this.constrictDuration || 0) + dt) : 0;
    this.huntTarget = (stance === 'HUNT') ? state.huntCandidate : null;
    this.huntDuration = (stance === 'HUNT') ? ((this.huntDuration || 0) + dt) : 0;

    const steerResult = computeSteering({
      stance,
      state,
      agent: this,
      snake,
      currentAng: state.currentAng,
      turnDelta,
      reflexSign,
      dangerUrgency,
      normWallDist: state.normWallDist,
      diffCenterAngle: state.diffCenterAngle,
      headRadius: state.headRadius,
      foodTarget,
      now,
      dt
    });
    turnDelta = steerResult.turnDelta;
    stance = steerResult.activeStance;
    this.combatStance = stance;

    if (dangerUrgency > 0.05) {
      this.fitness = Math.max(0, this.fitness - dt * 15.0 * dangerUrgency);
    }
    if (stance === 'HUNT' || stance === 'ATTACK' || stance === 'CONSTRICT') {
      const preyDist = state.huntCandidate ? state.huntCandidate.dist : (state.attackCandidate ? state.attackCandidate.distToIntercept : 999);
      const proximityBonus = Math.max(0, (600 - preyDist) / 600) * dt * 8.0;
      this.fitness += proximityBonus;
    }

    const smoothedTurn = turnDelta * 0.7 + this.prevTurnEma * 0.3;
    this.prevTurnEma = this.prevTurnEma * 0.82 + turnDelta * 0.18;
    this.targetAngle = state.currentAng + smoothedTurn * 1.85;

    if (this.boostCooldown > 0) {
      this.boostCooldown = Math.max(0, this.boostCooldown - dt);
    }
    if (this.boostFuel < 1) {
      this.boostFuel = Math.min(1, this.boostFuel + dt / this.boostRefillSeconds);
    }

    let shouldBoost = governBoost({
      stance,
      state,
      agent: this,
      snake,
      turnDelta,
      currentAng: state.currentAng,
      currentScore: state.currentScore,
      currentSpeed,
      dangerUrgency,
      normWallDist: state.normWallDist,
      headRadius: state.headRadius,
      foodTarget,
      boostProb
    });

    if (shouldBoost) {
      this.isBoosting = true;
      this.boostDuration += dt;
      this.boostFuel = Math.max(0, this.boostFuel - dt / this.boostBurnSeconds);
      if (this.boostDuration >= this.boostBurstSeconds || this.boostFuel <= 0) {
        this.isBoosting = false;
        this.boostDuration = 0;
        this.boostCooldown = this.boostRestSeconds;
      }
    } else {
      this.isBoosting = false;
      this.boostDuration = 0;
    }

    let dominantRayIdx = 8;
    let dominantRayDist = state.inputs[8];
    for (let r = 0; r < 16; r++) {
      if (state.inputs[r] > dominantRayDist) {
        dominantRayDist = state.inputs[r];
        dominantRayIdx = r;
      }
    }

    const activeLethalType = state.frontLethalType !== 'none' ? state.frontLethalType : state.closestLethalType;
    const activeLethalDist = state.frontLethalType !== 'none' ? state.frontLethalDist : state.minLethalDist;
    const activeLethalAngle = state.frontLethalType !== 'none' ? state.frontLethalAngle : state.closestLethalAngle;
    const activeLethalScale = state.frontLethalType !== 'none' ? state.frontLethalScale : state.closestLethalScale;
    const activeLethalSpeed = state.frontLethalType !== 'none' ? state.frontLethalSpeed : state.closestLethalSpeed;
    const lethalMassRatio = activeLethalScale / Math.max(0.01, scale);
    const activeThreatAngle = state.frontObstacleType !== 'none' ? state.frontObstacleAngle : state.closestObstacleAngle;

    this.flightRecorder.recordFrame({
      t: this.survivalTime,
      x: snake.xx ?? 0,
      y: snake.yy ?? 0,
      ang: state.currentAng,
      sp: currentSpeed,
      score: state.currentScore,
      headRadius: state.headRadius,
      len: (typeof snake.sct === 'number' && snake.sct > 0) ? snake.sct : Math.floor(state.currentScore / 10),
      stance: this.combatStance || 'FORAGE',
      normWallDist: state.normWallDist,
      threatDist: activeLethalType !== 'none' ? activeLethalDist : dangerSource,
      threatType: activeLethalType !== 'none' ? activeLethalType : (state.frontObstacleType !== 'none' ? state.frontObstacleType : state.closestObstacleType),
      threatAngle: activeLethalType !== 'none' ? activeLethalAngle : activeThreatAngle,
      threatMassRatio: activeLethalType !== 'none' ? lethalMassRatio : (this.massRatio || 1.0),
      threatSpeed: activeLethalType !== 'none' ? activeLethalSpeed : threatSpeed,
      turnDelta,
      isBoosting: this.isBoosting,
      selfShieldSide: state.selfShieldSide,
      dominantRayIdx,
      dominantRayDist
    });

    return {
      targetAngle: this.targetAngle,
      isBoosting: this.isBoosting,
      rawSteer: outputs[0],
      rawBoost: boostProb,
      fitness: Math.floor(this.fitness),
      bestFitness: Math.floor(this.bestFitness),
      generation: this.generation,
      currentEpisode: this.currentEpisode,
      score: state.currentScore,
      stance: this.combatStance,
      kills: this.kills,
      massRatio: Number(this.massRatio.toFixed(2)),
      threatScale: Number(this.threatScale.toFixed(2)),
      closestObstacleType: state.closestObstacleType,
      frontObstacleType: state.frontObstacleType,
      selfShieldSide: state.selfShieldSide,
      frontThreatSpeed: state.frontThreatSpeed,
      nearestThreatSpeed: state.nearestThreatSpeed,
      encirclementEscape: state.encirclementEscape,
      slipstreamTarget: state.slipstreamTarget,
      huntCandidate: state.huntCandidate
    };
  }

  learnFromHuman(snake, snakes, foods, preys, targetX, targetY, isHumanBoosting) {
    const now = performance.now();
    const dt = Math.min(0.1, Math.max(0.001, (now - this.lastDecisionTime) / 1000));
    this.lastDecisionTime = now;

    if (!snake || snake.dead) return;

    const survivalBonusRate = 4.0 + Math.min(6.0, (this.stagnantGens || 0) * 0.5);
    this.survivalTime += dt;
    this.fitness += dt * survivalBonusRate;
    this.touchedByHuman = true;

    const state = this._extractInputs(snake, snakes, foods, preys, dt, now);
    const { inputs, currentAng } = state;

    const humanAimAngle = Math.atan2(targetY, targetX);
    let diffAngle = humanAimAngle - currentAng;
    while (diffAngle < -Math.PI) diffAngle += Math.PI * 2;
    while (diffAngle > Math.PI) diffAngle -= Math.PI * 2;

    const humanSteerTarget = Math.max(-1.0, Math.min(1.0, diffAngle / 1.85));
    const humanBoostTarget = isHumanBoosting ? 1.0 : 0.0;

    const isTurning = Math.abs(humanSteerTarget) > 0.12;
    const isNearObstacle = state.minObstacleDist < state.headRadius * 8.0;
    const isCombatReflex = state.frontObstacleType === 'head' || state.closestObstacleType === 'head';
    const isTacticalLesson = isTurning || isNearObstacle || isHumanBoosting || isCombatReflex;
    const adaptiveLr = isCombatReflex ? 0.008 : (isTacticalLesson ? 0.006 : 0.0004);

    this.teacherSamples++;
    this.lifeTeacherSamples = (this.lifeTeacherSamples || 0) + 1;

    if (!this.teacherQuarantineBuffer) this.teacherQuarantineBuffer = [];
    this.teacherQuarantineBuffer.push({
      inputs: new Float32Array(inputs),
      targets: [humanSteerTarget, humanBoostTarget],
      lr: adaptiveLr,
      momentum: 0.80
    });

    if (this.teacherQuarantineBuffer.length > (this.teacherQuarantineCapacity || 45)) {
      const safeFrame = this.teacherQuarantineBuffer.shift();
      this.brain.trainStep(safeFrame.inputs, safeFrame.targets, safeFrame.lr, safeFrame.momentum);
    }

    this.prevTurnEma = this.prevTurnEma * 0.85 + humanSteerTarget * 0.15;
    this.targetAngle = humanAimAngle;
    this.isBoosting = isHumanBoosting;

    let dominantRayIdx = 0;
    let dominantRayDist = 0;
    for (let r = 0; r < this.rayCount; r++) {
      if (inputs[r] > dominantRayDist) {
        dominantRayDist = inputs[r];
        dominantRayIdx = r;
      }
    }

    const currentScore = (typeof snake.sct === 'number' && snake.sct > 0)
      ? snake.sct
      : (state.currentScore || (snake.sc ? Math.floor(15 * (snake.sc - 1) / 0.001) : 0));
    const currentSpeed = (typeof snake.sp === 'number' && snake.sp > 0) ? snake.sp : (isHumanBoosting ? 14 : 5.9);

    const activeLethalType = (state.frontLethalType && state.frontLethalType !== 'none')
      ? state.frontLethalType
      : ((state.closestLethalType && state.closestLethalType !== 'none') ? state.closestLethalType : 'none');
    const activeLethalDist = (state.frontLethalType && state.frontLethalType !== 'none')
      ? state.frontLethalDist
      : ((state.closestLethalType && state.closestLethalType !== 'none') ? (state.minLethalDist ?? state.closestLethalDist ?? 2000) : (Number.isFinite(state.frontObstacleDist) ? state.frontObstacleDist : state.minObstacleDist));
    const activeLethalAngle = (state.frontLethalType && state.frontLethalType !== 'none')
      ? state.frontLethalAngle
      : ((state.closestLethalType && state.closestLethalType !== 'none') ? state.closestLethalAngle : (state.frontObstacleType !== 'none' ? state.frontObstacleAngle : state.closestObstacleAngle));
    const activeLethalScale = (state.frontLethalType && state.frontLethalType !== 'none')
      ? state.frontLethalScale
      : ((state.closestLethalType && state.closestLethalType !== 'none') ? state.closestLethalScale : (state.frontThreatScale || 1));
    const activeLethalSpeed = (state.frontLethalType && state.frontLethalType !== 'none')
      ? state.frontLethalSpeed
      : ((state.closestLethalType && state.closestLethalType !== 'none') ? state.closestLethalSpeed : (state.frontThreatSpeed || 5.9));
    const lethalMassRatio = activeLethalScale / Math.max(0.01, snake.sc || 1);

    this.flightRecorder.recordFrame({
      t: this.survivalTime,
      x: snake.xx ?? 0,
      y: snake.yy ?? 0,
      ang: currentAng,
      sp: currentSpeed,
      score: currentScore,
      headRadius: state.headRadius,
      len: (typeof snake.sct === 'number' && snake.sct > 0) ? snake.sct : Math.floor(currentScore / 10),
      stance: 'TEACHER',
      normWallDist: state.normWallDist,
      threatDist: activeLethalType !== 'none' ? activeLethalDist : (Number.isFinite(state.frontObstacleDist) ? state.frontObstacleDist : state.minObstacleDist),
      threatType: activeLethalType !== 'none' ? activeLethalType : (state.frontObstacleType !== 'none' ? state.frontObstacleType : state.closestObstacleType),
      threatAngle: activeLethalType !== 'none' ? activeLethalAngle : (state.frontObstacleType !== 'none' ? state.frontObstacleAngle : state.closestObstacleAngle),
      threatMassRatio: lethalMassRatio,
      threatSpeed: activeLethalSpeed,
      turnDelta: humanSteerTarget,
      isBoosting: isHumanBoosting,
      selfShieldSide: state.selfShieldSide || 0,
      dominantRayIdx,
      dominantRayDist
    });
  }

  onDeath(isHumanDriven = false) {
    const runFitness = Math.max(0, this.fitness);
    const wasHumanAssisted = isHumanDriven || this.touchedByHuman;

    this.flightRecorder.deduceDeathAutopsy({
      finalScore: this.previousScore || 0,
      peakLength: this.peakLength || 0,
      survivalTime: this.survivalTime || 0,
      isHumanDriven: wasHumanAssisted,
      generation: this.generation,
      episode: this.currentEpisode
    });

    if (!this.abandonCurrentEpisode) {
      if (wasHumanAssisted) {
        this._queueTeacherCandidate(isHumanDriven, runFitness);
      } else {
        this._recordEvaluationEpisode(runFitness);
      }
    }

    this.flightRecorder.resetLife();
    this.fitness = 0;
    this.survivalTime = 0;
    this.previousScore = 0;
    this.peakLength = 0;
    this.enemyTracker.clear();
    this.prevTurnEma = 0;
    this.leftThreatEma = 0;
    this.rightThreatEma = 0;
    this.boostDuration = 0;
    this.boostCooldown = 0;
    this.boostFuel = 1;
    this.foodTargetError = null;
    this.touchedByHuman = false;
    this.abandonCurrentEpisode = false;
    this.coilSign = 0;
    this.lifeTeacherSamples = 0;
    this.closestObstacleType = 'none';
    this.frontObstacleType = 'none';
    this.selfShieldSide = 0;
    this.encirclementEscape = null;
    this.slipstreamTarget = null;
    this.evasionSign = 0;
    this.evasionLatchTimer = 0;
    this.escapeCorridorAngle = null;
    this.corridorTurnSign = 0;
    this.teacherQuarantineBuffer = [];
  }

  _queueTeacherCandidate(humanDriven, runFitness) {
    const MIN_TEACHER_SECONDS = 15;
    const MIN_TEACHER_SAMPLES = 300;
    const longEnough = this.survivalTime >= MIN_TEACHER_SECONDS;
    const samplesThisLife = this.lifeTeacherSamples > 0
      ? this.lifeTeacherSamples
      : (this.lastLoadedTimestamp > 0 ? 0 : this.teacherSamples);
    const sawEnoughTuition = samplesThisLife >= MIN_TEACHER_SAMPLES;

    if (!humanDriven || !longEnough || !sawEnoughTuition) {
      if (this.championBrain) {
        this.brain = NeuralNetwork.fromJSON(this.championBrain.export(), this.inputCount);
      }
      this.candidatePending = false;
      this.candidateSource = null;
      return;
    }

    this.candidatePending = true;
    this.candidateSource = 'teacher';
    this.currentEpisode = 1;
    this.episodeFitnesses = [];
    this._saveCheckpoint();
  }

    _recordEvaluationEpisode(runFitness) {
      // A life that was partly human-driven is not a fair sample of this
      // genome's ability, so it is dropped instead of being scored.
      if (this.abandonCurrentEpisode) {
        console.log('%c[SlitherML] ⏭️ Life excluded from the rollout (was partly human-driven, so it would bias this genome\'s score).', 'color: #94a3b8;');
        return;
      }

      // Mulligan filter: Server spawn-kills (instant collision on re-entry before snake can maneuver)
      // are unavoidable noise that poison 3-life rollouts. Exclude them from the rollout.
      const isSpawnKill = this.survivalTime > 0 && (
        (this.survivalTime < 3.5 && runFitness < 60) ||
        (this.survivalTime < 5.0 && runFitness < 40 && (!this.peakLength || this.peakLength <= 10))
      );
      if (isSpawnKill) {
        console.log(`%c[SlitherML] 🔄 Spawn-kill detected (${this.survivalTime.toFixed(1)}s, fitness ${Math.floor(runFitness)}, length ${this.peakLength || 0}). Mulligan granted: life excluded from rollout.`, 'color: #94a3b8;');
        return;
      }

      this.episodeFitnesses.push(runFitness);

      if (this.currentEpisode < this.episodesPerGenome) {
        this.currentEpisode++;
        this._saveCheckpoint();
        return;
      }

      // If fewer than 2 valid runs were recorded due to spawn-kill mulligans:
      if (this.episodeFitnesses.length < 2) {
        console.log(`%c[SlitherML] 🔄 Insufficient rollout data (${this.episodeFitnesses.length}/3 valid episodes due to spawn-kill mulligans). Extending rollout to gather reliable signal...`, 'color: #94a3b8;');
        this.currentEpisode = Math.max(1, this.episodeFitnesses.length + 1);
        this._saveCheckpoint();
        return;
      }

      const sorted = [...this.episodeFitnesses].sort((a, b) => b - a);
      const aggregateFitness = (sorted[0] + sorted[1]) / 2;
      const meanFitness = this.episodeFitnesses.reduce((a, b) => a + b, 0) / this.episodeFitnesses.length;

      // Track measured rollout distribution (rolling history of up to 20 rollouts)
      this.recentRollouts.push(aggregateFitness);
      if (this.recentRollouts.length > 20) this.recentRollouts.shift();

      let measuredMedian = null;
      let measuredQ3 = null;
      let measuredFloor = null;
      if (this.recentRollouts.length >= 2) {
        const sortedHistory = [...this.recentRollouts].sort((a, b) => a - b);
        measuredMedian = sortedHistory[Math.floor(sortedHistory.length / 2)];
        const q3Idx = Math.min(sortedHistory.length - 1, Math.floor(sortedHistory.length * 0.75));
        measuredQ3 = sortedHistory[q3Idx];
        measuredFloor = Math.max(500, measuredQ3);
      }

      const isTeacherCandidate = this.candidatePending && this.candidateSource === 'teacher';
      const origin = this.candidatePending ? 'teacher candidate' : 'mutant';
      const previousBest = this.bestFitness;
      // Two-Tier Tuition Preservation:
      // Tier 2: Candidate achieves >= 35% of champion, >= 1,500 points, or >= 1.4x measured median -> blend 20% into championBrain.
      // Tier 3: Candidate achieved rollout under Tier 2 -> championBrain is protected, but 15% is blended
      //         into baseBrain (mutant breeding seed) so valuable human imitation learning never evaporates.
      const teacherWorthyOfChampBlend = isTeacherCandidate && (aggregateFitness >= previousBest * 0.35 || aggregateFitness >= 5000);
      const teacherWorthyOfBreedingBlend = isTeacherCandidate && !teacherWorthyOfChampBlend && aggregateFitness <= previousBest;
      const isAutonomousMutant = !this.candidatePending;

      // Consistency gate: To prevent 1 lucky fluke run from crowning an erratic genome,
      // the median run (sorted[1]) must be at least 40% of previousBest.
      const passesConsistencyGate = sorted[1] >= previousBest * 0.40;

      // Mutant soft trait retention: autonomous mutants scoring >= 40% of previousBest
      // blend 15% weights into the baseBrain breeding seed (championBrain untouched).
      const mutantWorthyOfBlend = isAutonomousMutant && aggregateFitness <= previousBest && aggregateFitness >= previousBest * 0.40;

      // Prodigy Single-Run Rule: An autonomous mutant that achieved a heroic single life
      // (>= 22,000 points or >= 55% of all-time peak fitness) but missed the 40% aggregate bar
      // due to bad spawn-luck on remaining runs should NOT have its breakthrough weights discarded.
      // Blends 20% of its weights into the breeding template.
      const isProdigyRun = isAutonomousMutant && !mutantWorthyOfBlend && aggregateFitness <= previousBest &&
        sorted[0] >= Math.min(22000, (this.allTimePeakFitness || previousBest) * 0.55);

      // Strict elitism with Consistency Gate:
      if (aggregateFitness > this.bestFitness && passesConsistencyGate) {
        this.stagnantGens = 0;
        this.bestFitness = aggregateFitness;
        this.allTimePeakFitness = Math.max(this.allTimePeakFitness || 0, aggregateFitness);
        // Demote current champion to silver before crowning the new one
        if (this.championBrain) {
          this.silverBrain = NeuralNetwork.fromJSON(this.championBrain.export(), this.inputCount);
          this.silverFitness = previousBest;
          this.silverSource = 'champion';
        }
        this.championBrain = NeuralNetwork.fromJSON(this.brain.export(), this.inputCount);
        console.log(`%c[SlitherML] 🏆 NEW CHAMPION (${origin})! Gen ${this.generation} — Best-2-of-3 aggregate ${Math.floor(aggregateFitness)} (mean ${Math.floor(meanFitness)}, median ${Math.floor(sorted[1])}), previous ${Math.floor(previousBest)}`, 'color: #00f5d4; font-weight: bold;');
      } else {
        if (aggregateFitness > this.bestFitness && !passesConsistencyGate) {
          console.log(`%c[SlitherML] ⚠️ Candidate aggregate ${Math.floor(aggregateFitness)} exceeded champion ${Math.floor(this.bestFitness)} but was rejected by Consistency Gate (median ${Math.floor(sorted[1])} < 40% of champ ${Math.floor(previousBest * 0.40)}).`, 'color: #ffb703;');
        }
        this.stagnantGens = (this.stagnantGens || 0) + 1;
        // Plateau relief: after 6 consecutive stagnant generations, gently ease bar by 0.5%
        // (accelerating to 1.0% when stagnantGens >= 15) to prevent 1-in-1000 lucky spawn
        // outliers from permanently freezing evolution.
        // Adaptive Plateau Floor: Standard floor is 85% of peak fitness. When deep stagnation persists
        // (>= 50 gens without a champion), ease to 75%, and at >= 60 gens ease to 70% or toward measured distribution
        // to prevent infinite stagnation traps (e.g., Gen 134 at 67 stagnant gens or Gen 536 at 469 stagnant gens).
        this.allTimePeakFitness = Math.max(this.allTimePeakFitness || 0, this.bestFitness);
        const floorRatio = this.stagnantGens >= 60 ? 0.70 : (this.stagnantGens >= 50 ? 0.75 : 0.85);
        let minBar = this.allTimePeakFitness * floorRatio;
        if (this.stagnantGens >= 60) {
          const targetFloor = measuredFloor !== null ? measuredFloor : 500;
          if (minBar > targetFloor) {
            minBar = Math.max(targetFloor, minBar * Math.pow(0.985, this.stagnantGens - 60));
          }
        }
        if (this.stagnantGens >= 15) {
          const decayed = Math.max(minBar, this.bestFitness * 0.990);
          console.log(`%c[SlitherML] 📉 Deep Plateau Relief (Stagnant Gen ${this.stagnantGens}): Fitness bar eased from ${this.bestFitness.toFixed(1)} to ${decayed.toFixed(1)} (-1.0%${decayed === minBar ? `, capped at ${Math.round(floorRatio * 100)}% floor` : ''})`, 'color: #94a3b8;');
          this.bestFitness = decayed;
        } else if (this.stagnantGens >= 6) {
          const decayed = Math.max(minBar, this.bestFitness * 0.995);
          console.log(`%c[SlitherML] 📉 Plateau Relief (Stagnant Gen ${this.stagnantGens}): Fitness bar gently eased from ${this.bestFitness.toFixed(1)} to ${decayed.toFixed(1)} (-0.5%${decayed === minBar ? `, capped at ${Math.round(floorRatio * 100)}% floor` : ''})`, 'color: #94a3b8;');
          this.bestFitness = decayed;
        }

        // Fallback Silver Candidate: When no real displaced champion exists (or current silver
        // is also a mutant candidate), remember the best qualifying non-crowned autonomous mutant
        // (scoring >= 40% of champion, or achieving a prodigy run) as the silver candidate for genetic crossover.
        if ((mutantWorthyOfBlend || isProdigyRun) && this.silverSource !== 'champion') {
          if (!this.silverBrain || aggregateFitness > (this.silverFitness || 0)) {
            this.silverBrain = NeuralNetwork.fromJSON(this.brain.export(), this.inputCount);
            if (this.silverBrain && this.silverBrain.biases && this.silverBrain.biases.length > 0) {
              const sOutB = this.silverBrain.biases[this.silverBrain.biases.length - 1];
              if (sOutB && sOutB.length > 0) sOutB[0] = Math.max(-0.15, Math.min(0.15, sOutB[0]));
            }
            this.silverFitness = aggregateFitness;
            this.silverSource = 'mutant';
            console.log(`%c[SlitherML] 🥈 Silver candidate adopted: autonomous mutant scoring ${Math.floor(aggregateFitness)} (>= 40% of champ ${Math.floor(previousBest)}${isProdigyRun ? ' [Prodigy]' : ''}) saved for crossover recombination.`, 'color: #94a3b8; font-weight: bold;');
          }
        }

        if (this.candidatePending) {
          if (teacherWorthyOfChampBlend && this.championBrain) {
            console.log(`%c[SlitherML] 🧬 Teacher candidate achieved competitive rollout (${Math.floor(aggregateFitness)} vs ${Math.floor(this.bestFitness)}). Integrating 20% tuition into Champion template!`, 'color: #a855f7; font-weight: bold;');
            const candidateBrain = this.brain;
            for (let layer = 0; layer < this.championBrain.weights.length; layer++) {
              const chW = this.championBrain.weights[layer];
              const cW = candidateBrain.weights[layer];
              for (let j = 0; j < chW.length; j++) {
                chW[j] = (chW[j] * 0.8) + (cW[j] * 0.2);
              }
              const chB = this.championBrain.biases[layer];
              const cB = candidateBrain.biases[layer];
              for (let j = 0; j < chB.length; j++) {
                chB[j] = (chB[j] * 0.8) + (cB[j] * 0.2);
              }
            }
            if (this.championBrain.biases && this.championBrain.biases.length > 0) {
              const chOutB = this.championBrain.biases[this.championBrain.biases.length - 1];
              if (chOutB && chOutB.length > 0) {
                chOutB[0] = Math.max(-0.15, Math.min(0.15, chOutB[0]));
              }
            }
            this.stagnantGens = 0;
          } else {
            console.log(`%c[SlitherML] 🚫 Teacher candidate rejected from champion: aggregate ${Math.floor(aggregateFitness)} did not beat champion ${Math.floor(this.bestFitness)}. Champion genome preserved; infusing 15% tuition into mutant breeding template.`, 'color: #ffb703; font-weight: bold;');
          }
        }
      }

      this.generation++;
      this.candidatePending = false;
      this.candidateSource = null;

      // Breed the next genome with Anti-Stagnation Adaptive Mutation (from MEMWashburn/NEAT)
      // and High-Fitness Annealed Mutation (for elite champions > 30000 fitness).
      if (this.championBrain) {
        // Recombination: when stuck in a plateau for >= 5 gens and a silver brain
        // exists, breed via uniform crossover(champion, silver) instead of a pure
        // clone. This combines complementary sub-skills from two elite lineages and
        // explores regions of weight-space that point-mutation alone cannot reach.
        const useCrossover = this.stagnantGens >= 5 && this.silverBrain;
        const baseBrain = useCrossover
          ? NeuralNetwork.crossover(this.championBrain, this.silverBrain)
          : NeuralNetwork.fromJSON(this.championBrain.export(), this.inputCount);
        if (useCrossover) {
          console.log(`%c[SlitherML] 🧬 Recombination (Stagnant Gen ${this.stagnantGens}): breeding via crossover(champion, silver) to escape fitness plateau!`, 'color: #a855f7; font-weight: bold;');
        }
        if (baseBrain) {
          if ((mutantWorthyOfBlend || teacherWorthyOfBreedingBlend || isProdigyRun) && this.brain) {
            let blendRatio = 0.15;
            let blendReason = '';
            if (teacherWorthyOfBreedingBlend) {
              blendRatio = 0.15;
              blendReason = `Preserving human tuition (${Math.floor(aggregateFitness)} aggregate): Blending 15% teacher candidate weights into mutant breeding template!`;
            } else if (isProdigyRun) {
              blendRatio = 0.20;
              blendReason = `🌟 Prodigy single-run breakthrough (${Math.floor(sorted[0])} peak life, aggregate ${Math.floor(aggregateFitness)} vs champ ${Math.floor(this.bestFitness)}): Blending 20% weights into mutant breeding seed!`;
            } else {
              blendRatio = 0.15;
              blendReason = `Autonomous mutant showed breakthrough performance (${Math.floor(aggregateFitness)} aggregate vs champ ${Math.floor(this.bestFitness)}). Blending 15% weights into mutant breeding template!`;
            }
            console.log(`%c[SlitherML] 🧬 ${blendReason}`, 'color: #a855f7; font-weight: bold;');
            const candidateBrain = this.brain;
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

          const isStagnant = this.stagnantGens >= 3;
          const isDeepStagnant = this.stagnantGens >= 8;
          const isElite = this.bestFitness > 30000;

          let mutRate = isDeepStagnant ? 0.18 : (isStagnant ? 0.12 : 0.06);
          let mutStrength = isDeepStagnant ? 0.35 : (isStagnant ? 0.28 : 0.16);

          if (isElite) {
            mutRate = isDeepStagnant ? 0.09 : (isStagnant ? 0.06 : 0.035);
            mutStrength = isDeepStagnant ? 0.20 : (isStagnant ? 0.14 : 0.08);
          }

          if (isDeepStagnant) {
            console.log(`%c[SlitherML] 🔥 Deep Stagnation Shake (Gen ${this.generation}, Stagnant ${this.stagnantGens}): Escalating mutation to escape deep local minimum (rate ${mutRate}, strength ${mutStrength}${isElite ? ', annealed' : ''})!`, 'color: #f43f5e; font-weight: bold;');
          } else if (isStagnant) {
            console.log(`%c[SlitherML] ⚡ Anti-Stagnation Activated (Gen ${this.generation}): Escalating mutation (rate ${mutRate}, strength ${mutStrength}${isElite ? ', annealed' : ''}) to escape local minima!`, 'color: #a855f7; font-weight: bold;');
          }
          baseBrain.mutate(mutRate, mutStrength);
          this.brain = baseBrain;
        }
      } else {
        this.brain.mutate(0.08, 0.20);
      }

      this.currentEpisode = 1;
      this.episodeFitnesses = [];
      this._saveCheckpoint();
    }

    _saveChampionWeights() {
      try {
        const championExport = this.championBrain
          ? this.championBrain.export()
          : this.brain.export();
        const activeExport = this.brain.export();
        const data = {
          brain: championExport,
          activeBrain: activeExport,
          hasChampion: Boolean(this.championBrain),
          silverBrain: this.silverBrain ? this.silverBrain.export() : null,
          silverFitness: this.silverFitness || 0,
          silverSource: this.silverSource || null,
          bestFitness: this.bestFitness,
          allTimePeakFitness: this.allTimePeakFitness || this.bestFitness,
          generation: this.generation,
          currentEpisode: this.currentEpisode,
          episodeFitnesses: this.episodeFitnesses,
          candidatePending: this.candidatePending,
          candidateSource: this.candidateSource,
          kills: this.kills,
          teacherSamples: this.teacherSamples,
          allTimePeakLength: this.allTimePeakLength,
          stagnantGens: this.stagnantGens,
          telemetry: this.flightRecorder.getTelemetrySummary(),
          timestamp: Date.now()
        };
        const serialized = JSON.stringify(data);
        localStorage.setItem(STORAGE_KEY, serialized);
        if (typeof window.__slitherMLBridgeSave === 'function') {
          window.__slitherMLBridgeSave(serialized);
        }
        if (typeof GM_setValue === 'function') {
          try {
            GM_setValue(STORAGE_KEY, serialized);
          } catch (_) {}
        }
      } catch (e) {
        console.warn('[SlitherML] Could not persist checkpoint:', e && e.message ? e.message : e);
      }
    }


  getAutopsyReport() {
    return this.flightRecorder.lastDeathAutopsy;
  }

  exportCheckpoint() {
    const data = {
      brain: (this.championBrain || this.brain).export(),
      activeBrain: this.brain.export(),
      hasChampion: Boolean(this.championBrain),
      silverBrain: this.silverBrain ? this.silverBrain.export() : null,
      silverFitness: this.silverFitness || 0,
      silverSource: this.silverSource || null,
      bestFitness: this.bestFitness,
      allTimePeakFitness: this.allTimePeakFitness || this.bestFitness,
      generation: this.generation,
      currentEpisode: this.currentEpisode,
      episodeFitnesses: this.episodeFitnesses,
      candidatePending: this.candidatePending,
      candidateSource: this.candidateSource,
      kills: this.kills,
      teacherSamples: this.teacherSamples,
      allTimePeakLength: this.allTimePeakLength,
      stagnantGens: this.stagnantGens,
      telemetry: this.flightRecorder.getTelemetrySummary(),
      timestamp: Date.now()
    };
    return JSON.stringify(data, null, 2);
  }
}
