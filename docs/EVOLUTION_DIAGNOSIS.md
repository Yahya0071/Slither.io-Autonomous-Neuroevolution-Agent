# Evolutionary Stagnation Diagnosis & Telemetry Analysis (Phase 3)

## Executive Summary

Analysis of the lineage checkpoints (`slither_ml_brain_gen100.json` through `gen536.json`) reveals that autonomous evolution completely froze after generation ~67. At generation 536:
- `stagnantGens = 469` (zero evolutionary breakthroughs for 469 generations)
- `bestFitness = 30,543.8` (pinned immovably at exactly `0.70 × 43,634`, the plateau floor)
- `silverBrain = null` (recombination crossover never fired once in 469 generations)
- `VACUUM` stance duty cycle: **77.3%** (dominating 59%–93% across lives)
- Arena zone distribution: **100.0% center**
- Peak length across recent episodes: **11–23** (agent never grows into an adult snake)

The agent is trapped in a multi-system failure mode where heuristic overrides suppress neural policy learning, while an unreachable fitness floor prevents any mutant from ever being selected.

---

## 1. Root Cause Analysis

### Cause 1: The Human-Peak Fitness Floor Trap
- **How Peak Fitness was Established**: In `checkpoints/slither_ml_brain_gen100.json`, `allTimePeakFitness = 43,634` with 168 kills, 1.08M teacher lessons, and length 22,783. This peak was achieved during human Grandmaster play in Teacher Mode.
- **The Stagnation Floor Formula** (`console_inject.js:2956`):
  ```javascript
  const floorRatio = this.stagnantGens >= 60 ? 0.70 : (this.stagnantGens >= 50 ? 0.75 : 0.85);
  const minBar = (this.allTimePeakFitness || this.bestFitness) * floorRatio;
  ```
  At `stagnantGens >= 60`, `minBar = 43,634 × 0.70 = 30,543.8`. Plateau relief cannot ease `bestFitness` below this threshold.
- **The Unreachable Bar**: To crown a new champion, an autonomous newborn snake (score 2) must achieve `Aggregate > 30,543.8` across a 3-life rollout while passing the Consistency Gate (`Median >= 12,217.5`).
- **The Fitness Math in Live Play**:
  - Survival bonus during stagnation: `10.0` points/second (`4.0 + min(6.0, stagnantGens * 0.5)`).
  - Growth bonus: `25.0` points per unit score.
  - Danger penalty: up to `-15.0` points/second.
  - An autonomous snake scoring 15 with 0 kills produces `~500` to `~1,500` fitness per life (as documented in `gen536` episode history).
  - To reach `30,544` fitness at score 15 with 0 kills requires surviving **3,054 seconds (~51 minutes)** in a single life! In a public Slither.io lobby crowded with 400 aggressive player snakes, 51-minute survival without human evasion is statistically near-impossible.
- **Result**: 100% of autonomous mutants fail to crown. `stagnantGens` ticked upward monotonically to 469.

---

### Cause 2: Silver Brain & Crossover Deadlock
- **The Intended Mechanism**: Uniform crossover (`NeuralNetwork.crossover(champion, silver)`) was designed to break stagnation plateaus when `stagnantGens >= 5`.
- **The Bug**:
  - `silverBrain` adoption originally required the mutant to score `≥ 75%` of champion fitness (`0.75 × 30,544 = 22,908`).
  - Because autonomous mutants only score 500–2,000, not a single mutant in 469 generations ever met the 22,908 threshold.
  - As a result, `silverBrain` remained `null` indefinitely.
  - Line 3024:
    ```javascript
    const useCrossover = this.stagnantGens >= 5 && this.silverBrain;
    ```
    Evaluated to `false` on every single generation. Crossover **never executed once**.

---

### Cause 3: VACUUM Stance Monopolization (59%–93% Duty Cycle)
- **The Trigger** (`console_inject.js:2105, 2229`):
  ```javascript
  const richCorpseNearby = Boolean(foodTarget && (foodTarget.isCorpse || foodTarget.clusterCount >= 2) && foodTarget.dist < 850);
  ...
  else if (((this.lastKillEpicenter && now < this.vacuumActiveUntil) || richCorpseNearby) && dangerUrgency < 0.38) {
    stance = 'VACUUM';
  }
  ```
  - `foodTarget.isCorpse` is true for any pellet with `sz >= 6.5`.
  - `foodTarget.clusterCount >= 2` is true whenever 2 pellets are within 160 units of each other.
  - In Slither.io, large ambient glowing pellets and boost trails are ubiquitous within 850 units. Therefore, `richCorpseNearby` evaluates to `true` virtually 100% of the time.
- **The Architectural Damage**:
  1. **Neural Policy Muted by 95% Heuristic Assist** (`console_inject.js:2435`):
     ```javascript
     const assist = (stance === 'VACUUM' ? 0.95 : 0.85) * (1 - dangerUrgency) * safetyGating;
     turnDelta = turnDelta * (1 - assist) + this.foodSteerEma * assist;
     ```
     In VACUUM stance, heuristic food steering overrides 95% of the neural network's steering output. The neural network has no control over the snake's heading during foraging. Because the weights have almost zero influence on survival, evolutionary mutation produces no meaningful selective gradient.
  2. **The "Left-Right" Yaw Oscillation Trap** (`console_inject.js:2427-2430`):
     ```javascript
     else if (stance === 'VACUUM' && foodTarget.value >= 8 && foodTarget.dist > 120 && foodTarget.dist < 600) {
       const crossOffset = foodTarget.error >= 0 ? 0.20 : -0.20;
       foodSteer = Math.max(-1, Math.min(1, (foodTarget.error / 1.85) + crossOffset));
     }
     ```
     "The Cross" perpendicular loot sweep forces alternating `+0.20` and `-0.20` yaw offsets. When surrounded by multiple threats, the agent remains in VACUUM (because pellets are present), oscillating violently left and right until an enemy body cuts it off.

---

### Cause 4: 100% Center-Zone Death Trap
- **Observation**: `gen536` telemetry reports `zoneDistribution.center: 100.0%`.
- **Mechanism**: Goldilocks Ring Guidance and Channel 31 center-bias pull the snake toward `(0, 0)`.
- **Effect**: The arena center is the highest-density combat zone in Slither.io. Newborn worms (`score 2..15`) spawned into or navigating directly into the center collide with giant snakes within 30–60 seconds, resulting in repetitive `BODY_COLLISION` and `ENCIRCLED_TRAP` autopsies.

---

## 2. Proposed Fixes & Verification Strategy

| Problem | Proposed Fix | Verification Test |
| :--- | :--- | :--- |
| **1. Unreachable Plateau Floor** | Decouple the autonomous evaluation floor from human teacher records: When `stagnantGens >= 15`, allow `bestFitness` decay to ease down to `1.5×` the rolling maximum of recent autonomous rollouts, preventing an unreachable human score from freezing evolution. | Unit test verifying that an autonomous lineage with `stagnantGens >= 15` eases its bar toward achievable autonomous levels. |
| **2. Silver Brain Starvation** | Guarantee a silver candidate exists during stagnation: When `stagnantGens >= 5` and no candidate has qualified via the 40% threshold, adopt the best-scoring mutant from the last generation rollout as silver candidate. | Unit test verifying `silverBrain` is populated and `NeuralNetwork.crossover` executes at `stagnantGens >= 5`. |
| **3. VACUUM Stance Hijack** | Require authentic corpse density: Change `richCorpseNearby` to require `(foodTarget.isCorpse && foodTarget.clusterCount >= 5) || foodTarget.value >= 25 || (this.lastKillEpicenter && now < this.vacuumActiveUntil)`. Ordinary ambient pellets must use `FORAGE` stance. | Unit test asserting that ambient pellets with clusterCount = 2 trigger `FORAGE`, while high-density corpse clusters trigger `VACUUM`. |
| **4. Heuristic Over-Assist in Foraging** | Lower `assist` in foraging from 0.95 to 0.40–0.50 so the neural network's steering policy actually influences trajectory and experiences selective pressure. | Simulation benchmark test measuring variance in path efficiency between distinct neural genomes in foraging fields. |
