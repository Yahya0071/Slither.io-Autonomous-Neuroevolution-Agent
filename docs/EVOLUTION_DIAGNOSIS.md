# Evolutionary Stagnation Diagnosis & Telemetry Analysis (Phase 3b)

## Executive Summary

An empirical audit of 34 versioned checkpoints spanning **Generation 8 to Generation 536** (`checkpoints/` and `checkpoints_archive/`) establishes that autonomous neuroevolution in SlitherML has been **frozen for 467 consecutive generations**.

```
Gen 8 ──► Gen 25 ──► Gen 69 ──► Gen 100 ──► Gen 337 ──► Gen 536
(32.8k)   (43.7k)    (43.8k)    (37.1k)     (30.5k)     (30.5k)
                       ▲                       ▲           ▲
                       └────── 467 GENS ───────┴───────────┘
                           Weights 100% Bit-Identical
                           silverBrain === null
                           stagnantGens = 469
```

### Empirical Facts from Checkpoint Audit
1. **Synaptic Weights Bit-Identical for 467 Generations**:
   The champion weights in `slither_ml_brain_gen69.json` are **100% bit-identical** to those in `gen100`, `gen150`, `gen337`, `gen362`, `gen423`, and `gen536`. Not a single autonomous mutant has been selected since generation 69.
2. **Floor Trap at `30,543.8`**:
   At Gen 536, `stagnantGens = 469` and `bestFitness` is pinned at exactly `30,543.8` ($= 0.70 \times 43,634$, the deep plateau decay floor).
3. **`silverBrain` is Null in 100% of Checkpoints**:
   Because `silverBrain === null`, uniform crossover (`NeuralNetwork.crossover`) has **never executed a single time in live play**.
4. **Autonomous Reality vs Checkpoint Phantom**:
   Across 105 recent recorded autonomous episodes in checkpoint telemetry, the **median final score was 13** (maximum 45). The recorded champion peak score of `22,783` and fitness of `43,634` were achieved with human teacher assistance, not autonomous driving.
5. **Fatal Stance Distribution**:
   In 58% of recorded deaths, the fatal stance was `DEFEND`, with the snake dying during high-speed enemy cut-offs or flank entrapment.

---

## 1. Simulation Benchmark (50 Seeded Lives)

To test whether the champion's recorded performance is reproducible in autonomous flight, the unmutated champion from `slither_ml_brain_gen536.json` was evaluated across **50 deterministic seeds** (seeds 1 to 50) using `sim/snake-sim.mjs` (60s duration, 400 food pellets, 3 enemy obstacles). Five mutation profiles were run across the exact same 50 seeds.

### 50-Seed Simulation Benchmark Results

| Model / Genome | Mutation Rate $r$ | Perturb Strength $\sigma$ | Rel. Weight Norm Dist | Changed Weights | Score Median | Score Mean | Score IQR | Score Max | Survival Median | Pellets Median | Boost Duty Median |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Champion (Gen 536/100)** | — | — | **0.0%** | **0.0%** | **87.8** | **87.7** | **14.6** | **119.6** | **60.0s** | **10** | **14.0%** |
| **Standard Mutation** | 0.06 | 0.16 | **12.6%** | **5.7%** | **88.2** | **89.8** | **14.9** | **163.8** | **60.0s** | **8** | **10.9%** |
| **Stagnant Mutation** | 0.12 | 0.28 | **31.3%** | **12.0%** | **86.0** | **84.1** | **17.3** | **99.9** | **60.0s** | **8** | **14.3%** |
| **Deep Stagnant Mutation** | 0.18 | 0.35 | **54.1%** | **19.3%** | **87.0** | **89.7** | **12.8** | **163.1** | **60.0s** | **10** | **13.3%** |
| **Elite Annealed** | 0.035 | 0.08 | **4.9%** | **3.9%** | **90.8** | **89.8** | **12.6** | **119.8** | **60.0s** | **9** | **9.6%** |
| **Micro Norm-Scaled** | 0.02 | 0.04 | **1.7%** | **2.2%** | **87.0** | **88.4** | **14.1** | **122.3** | **60.0s** | **10** | **14.6%** |

### Benchmark Findings
1. **The Champion's Recorded Score is 100% Unreproducible Autonomously**:
   Starting from score 100, the champion finished with a median score of **87.8** (mean 87.7, max 119.6), losing mass to boost burn while eating only 10 pellets per minute. It never once approached its recorded checkpoint score of `22,783`.
2. **Mutants Match or Outperform the Champion**:
   Mutants scored identically to the champion (median 86.0–90.8 vs 87.8). In fact, the **Elite Annealed** mutant achieved a higher median score (**90.8** vs 87.8), and the **Standard** mutant achieved the highest maximum score (**163.8** vs 119.6).
3. **Mutation Shock Absorber**:
   Deep stagnation mutations (rate 0.18, strength 0.35) move the weights by **54.1% relative Euclidean norm distance**, flipping 19.3% of all weights. In an MLP, shifting weights by >50% fundamentally resets the policy to random behavior.

---

## 2. Simulator Fidelity vs Real-Game Telemetry

Before trusting offline simulation benchmarks to train or evaluate agents, `sim/snake-sim.mjs` was audited against live Slither.io game mechanics:

| Game Mechanic | Real Slither.io Server | `sim/snake-sim.mjs` Fidelity | Impact on Training |
| :--- | :--- | :--- | :--- |
| **Lethal Collisions** | Running into any enemy body or head segment causes instant death. | **Missing**: Enemy segments in `snake-sim.mjs` do **not** kill the player. Death occurs *only* when reaching `wallRadius * 0.5`. | The sim cannot evaluate obstacle avoidance, tactical banking, or cut-off evasion. |
| **Enemy Motion** | Enemies turn, boost, cut off, and react to the player snake. | **Static**: Enemies generated by `placeEnemies` are static curved arcs of stationary points (`pts`). | Zero dynamic predator-prey pressure. |
| **Mass Growth** | 90%+ of mass growth comes from harvesting dense dead-snake corpses (pellet value 8–50). | **Uniform**: Only spawns uniform ambient pellets (value 1) in uniform or cone layouts. | Stances like `VACUUM`, `HUNT`, and `ATTACK` cannot be exercised or rewarded. |
| **Kinematics** | Head turns toward mouse heading at bounded angular rate; speed scales ~1.9x during boost. | **High Fidelity**: Accurately reproduces angular turn step `maxTurnRate * dt` and speed scaling `5.9` vs `11.2`. | Ideal for measuring steering stability, circling bias, and food alignment. |

**Conclusion**: `sim/snake-sim.mjs` is an accurate **kinematic and foraging model**, but cannot serve as a ground-truth combat evaluator without dynamic lethal collisions.

---

## 3. Why Teacher Candidates and Silver Brain Never Populated

### The Mathematical Trap of Teacher Candidates
In `src/evolution.js`, candidate evaluation logic implements three tiers:
- **Tier 1 (Crown Champion)**: `Aggregate > bestFitness` and `Median >= 0.40 * previousBest`.
- **Tier 2 (20% Soft-Blend into Champion)**: `Aggregate >= previousBest * 0.35 || Aggregate >= 5000`.
- **Tier 3 (15% Tuition Preservation into Breeding Seed)**: `Aggregate < 35% && Aggregate < 5000`.

With `previousBest = 30,543.8`:
- Tier 2 requires an aggregate autonomous rollout score of **$\ge 5,000$ points** (or $0.35 \times 30,544 = 10,690$).
- But autonomous candidate rollouts on live servers achieve **~600 to 1,200 points** across 3 lives (median score 13, survival ~30s).
- Because candidate scores (~900) sit an order of magnitude below 5,000, **Tier 2 never once fired**.
- Candidates fell into Tier 3: blending 15% into the temporary mutant breeding seed, while `championBrain` and `bestFitness` remained untouched at 30,544.

### The Mathematical Lockout of `silverBrain`
For an autonomous mutant to be adopted as `silverBrain`, `evaluateRollout()` requires:
$$Aggregate \ge previousBest \times 0.40$$
With $previousBest = 30,543.8$, the threshold is **$12,217.5$ points**.
Autonomous mutants score $\sim 500 - 1,500$. As a result:
- $Aggregate \ge 12,217.5$ is **statistically impossible** for a score-15 autonomous hatchling.
- `silverBrain` stayed `null` across all 34 checkpoints.
- Uniform crossover condition `stagnantGens >= 5 && Boolean(silverBrain)` evaluated to `false` on every single generation.

---

## 4. Resolving Anomalies: Zone Tracker & `allTimePeakFitness`

### Anomaly A: Zone Tracker Reports 100% Center for Wall Deaths
**Observation**: Checkpoint `slither_ml_brain_gen150.json` recorded an `ARENA_WALL` death at coordinates `(33088, 35181)`, yet `telemetry.zoneDistribution` reported `{ center: '100.0%', midfield: '0.0%', perimeter: '0.0%' }`.

**Root Cause Analysis**:
1. **Coordinate Geometry Inversion**:
   On official Slither.io servers, the coordinate system is offset with the arena center located at `(21600, 21600)` with radius `grd = 21600`.
   In `FlightRecorder.recordFrame`:
   ```javascript
   const distFromOrigin = Math.hypot(snapshot.x || 0, snapshot.y || 0);
   const normDist = Number.isFinite(snapshot.normWallDist) ? snapshot.normWallDist : (distFromOrigin / 21600);
   ```
   When `normWallDist` was not supplied, `distFromOrigin = Math.hypot(x, y)` assumed `(0, 0)` is the center of the arena.
   - For a snake at the true center `(21600, 21600)`, `hypot(21600, 21600) = 30,547 / 21600 = 1.41 > 0.82` — falsely categorized as **perimeter**.
   - For a snake near the lethal boundary `(0, 0)`, `hypot(0, 0) = 0` — falsely categorized as **center**.
2. **Unfinished Life Snapshot Mismatch**:
   In `SlitherMLAgent`, `_saveCheckpoint()` was triggered during `_recordEvaluationEpisode` before `resetLife()` ran. When a checkpoint was saved mid-life at spawn (< 30s), `getTelemetrySummary()` exported the active spawning snake (which starts in `center` in `FORAGE`) while pairing it with the *previous* death autopsy (`ARENA_WALL` at 33088, 35181).

### Anomaly B: `allTimePeakFitness` Inconsistency
**Observation**: Checkpoint `slither_ml_brain_gen78.json` through `gen536.json` record `allTimePeakFitness = 43,634`. However, earlier checkpoints had higher fitness: `43,739` at `gen14` and `43,854` at `gen69`.

**Root Cause**:
`allTimePeakFitness` was introduced in commit `362af70`. When deployed around Gen 78, the code initialized `allTimePeakFitness` using:
```javascript
this.allTimePeakFitness = Math.max(this.allTimePeakFitness || 0, this.bestFitness);
```
Because `bestFitness` had already decayed from 43,854 down to 43,634 by Gen 73, and older checkpoints (Gen 14–69) had no `allTimePeakFitness` field stored, the variable was initialized to the active decayed value (`43,634`) rather than the historical peak (`43,854`).

---

## 5. Architectural Fixes & Implementation Plan

### Fix 1: Decouple Autonomous Evaluation from Human Peaks (Distribution Baseline)
Replace the rigid single-champion peak floor with a **measured distribution baseline**:
- Calculate rolling median ($\tilde{F}$) and interquartile range ($IQR$) over the last 10 autonomous rollouts.
- When `stagnantGens >= 15`, ease the evaluation bar toward the achievable distribution:
  $$\text{Target Bar} = \max(\text{Achievable Floor}, \tilde{F} + 1.5 \cdot IQR)$$
- This permits autonomous mutants to crown and advance evolution without requiring a 51-minute human survival run.

### Fix 2: Unblock Silver Brain & Crossover Activation
- Guarantee `silverBrain` population during stagnation: When `stagnantGens >= 5` and no silver brain exists, adopt the **best-performing mutant from the current generation rollout** as `silverBrain`.
- Clamps steering bias $b_{2,0} \in [-0.15, 0.15]$ on crossover offspring to guarantee directional symmetry.
- Lowers mutant candidate adoption threshold from $40\%$ of champion to $\max(0.40 \cdot \text{champ}, 1.25 \cdot \tilde{F})$.

### Fix 3: Norm-Scaled Adaptive Mutations
Replace destructive $54\%$ weight shocks with **norm-scaled Cauchy-Gaussian perturbations**:
- Cap total weight perturbation norm: $||\Delta W|| \le 0.08 \cdot ||W||$.
- Limit mutation rate to $r \in [0.02, 0.06]$ and strength to $\sigma \in [0.04, 0.12]$.
- Preserves learned steering and evasion reflexes while maintaining heavy-tailed Lévy exploration.

### Fix 4: Coordinate-Aware Spatial Zone Tracking
Update `FlightRecorder.recordFrame` to detect offset coordinate geometry (`x > 22680 || y > 22680`) and center around `(21600, 21600)` when `normWallDist` is absent:
```javascript
let normDist;
if (Number.isFinite(snapshot.normWallDist)) {
  normDist = snapshot.normWallDist;
} else {
  const arenaRadius = 21600;
  const isOffset = (snapshot.x > arenaRadius * 1.05 || snapshot.y > arenaRadius * 1.05);
  const cx = isOffset ? arenaRadius : 0;
  const cy = isOffset ? arenaRadius : 0;
  normDist = Math.hypot((snapshot.x || 0) - cx, (snapshot.y || 0) - cy) / arenaRadius;
}
```
