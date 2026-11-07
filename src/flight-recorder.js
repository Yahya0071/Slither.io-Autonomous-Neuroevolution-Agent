/**
 * Flight Recorder & Forensic Deductor
 * 60-frame rolling FIFO buffer, death root cause classification,
 * zone distributions, stance duty cycles, and pathfinder breadcrumbs.
 */

export class FlightRecorder {
  constructor(options = {}) {
    this.bufferCapacity = options.bufferCapacity || 60; // 60 frames (~1.0s at 60 FPS)
    this.frameBuffer = [];
    this.bufferIndex = 0;

    // Compact Pathfinder Breadcrumbs
    this.maxWaypoints = 60;
    this.waypointIntervalSec = 2.5;
    this.lastWaypointTime = -999;
    this.waypoints = []; // [[x, y, Math.round(t), Math.round(score), stance], ...]
    this.lastLifeWaypoints = [];

    // Growth Milestones
    this.milestoneTargets = [500, 1000, 2500, 5000, 10000, 15000, 20000];
    this.growthMilestones = {}; // { "500": 14.2, "1000": 32.5, ... }

    // Spatial Zone Distribution (frame counters)
    this.zoneCounts = { center: 0, midfield: 0, perimeter: 0 };
    this.totalFrames = 0;

    // Stance Frame Counters
    this.stanceCounts = {
      FORAGE: 0,
      ATTACK: 0,
      COIL: 0,
      CONSTRICT: 0,
      DEFEND: 0,
      VACUUM: 0,
      HUNT: 0,
      TEACHER: 0
    };

    this.lastLifeFrames = 0;
    this.lastLifeDutyCycles = {};
    this.lastLifeZones = { center: '0%', midfield: '0%', perimeter: '0%' };
    this.lastLifeMilestones = {};

    // Significant Incidents Log (max 25 per life)
    this.maxIncidents = 25;
    this.incidents = [];

    // Autopsy & Episode History
    this.lastDeathAutopsy = null;
    this.recentEpisodes = []; // max 10
    this.maxRecentEpisodes = 10;
  }

  resetLife() {
    if (this.waypoints.length > 0) {
      this.lastLifeWaypoints = this.waypoints.slice(-40);
    }
    if (this.totalFrames > 0) {
      this.lastLifeFrames = this.totalFrames;
      this.lastLifeDutyCycles = this._calculateStanceDutyCycles();
      this.lastLifeZones = this._calculateZoneDistribution();
      this.lastLifeMilestones = { ...this.growthMilestones };
    }
    this.frameBuffer = [];
    this.bufferIndex = 0;
    this.waypoints = [];
    this.lastWaypointTime = -999;
    this.growthMilestones = {};
    this.zoneCounts = { center: 0, midfield: 0, perimeter: 0 };
    this.totalFrames = 0;
    for (const k of Object.keys(this.stanceCounts)) {
      this.stanceCounts[k] = 0;
    }
    this.incidents = [];
  }

  recordFrame(snapshot) {
    if (!snapshot) return;
    this.totalFrames++;

    // 1. Rolling FIFO Frame Buffer
    if (this.frameBuffer.length < this.bufferCapacity) {
      this.frameBuffer.push(snapshot);
    } else {
      this.frameBuffer[this.bufferIndex] = snapshot;
      this.bufferIndex = (this.bufferIndex + 1) % this.bufferCapacity;
    }

    // 2. Stance Counter
    const st = snapshot.stance || 'FORAGE';
    if (typeof this.stanceCounts[st] === 'number') {
      this.stanceCounts[st]++;
    } else {
      this.stanceCounts[st] = 1;
    }

    // 3. Spatial Zone Tracking (Arena radius grd is ~21,600)
    const distFromOrigin = Math.hypot(snapshot.x || 0, snapshot.y || 0);
    const normDist = Number.isFinite(snapshot.normWallDist) ? snapshot.normWallDist : (distFromOrigin / 21600);
    if (normDist < 0.45) {
      this.zoneCounts.center++;
    } else if (normDist < 0.82) {
      this.zoneCounts.midfield++;
    } else {
      this.zoneCounts.perimeter++;
    }

    // 4. Growth Milestones
    const currentScore = snapshot.score || 0;
    for (let i = 0; i < this.milestoneTargets.length; i++) {
      const target = this.milestoneTargets[i];
      if (currentScore >= target && typeof this.growthMilestones[target] !== 'number') {
        this.growthMilestones[target] = Math.round((snapshot.t || 0) * 10) / 10;
      }
    }

    // 5. Pathfinder Breadcrumbs (compact downsampled waypoints)
    const t = snapshot.t || 0;
    if (t - this.lastWaypointTime >= this.waypointIntervalSec) {
      this.lastWaypointTime = t;
      if (this.waypoints.length >= this.maxWaypoints) {
        this.waypoints.shift();
      }
      this.waypoints.push([
        Math.round(snapshot.x || 0),
        Math.round(snapshot.y || 0),
        Math.round(t),
        Math.round(currentScore),
        snapshot.stance || 'FORAGE'
      ]);
    }
  }

  logIncident(type, details = {}) {
    if (this.incidents.length >= this.maxIncidents) return;
    this.incidents.push({
      t: Math.round((details.t || 0) * 10) / 10,
      type,
      x: Math.round(details.x || 0),
      y: Math.round(details.y || 0),
      info: details.info || ''
    });
  }

  deduceDeathAutopsy(meta = {}) {
    const orderedFrames = this._getOrderedFrames();
    const last = orderedFrames[orderedFrames.length - 1] || null;
    const prev = orderedFrames.length > 5 ? orderedFrames[orderedFrames.length - 5] : last;

    let cause = 'SUDDEN_IMPACT';
    let description = 'Sudden collision or server despawn';
    let confidence = 0.5;

    if (last) {
      const estScale = last.score ? Math.min(3.5, 1 + last.score / 15000) : 1;
      const headRadius = (typeof last.headRadius === 'number' && last.headRadius > 0)
        ? last.headRadius
        : (14 * estScale);
      const nearWall = (typeof last.normWallDist === 'number')
        ? (last.normWallDist >= 0.94 || (last.threatType === 'wall' && last.threatDist <= headRadius * 4.0))
        : (Math.hypot(last.x || 0, last.y || 0) >= 21600 - 350);

      if (nearWall) {
        cause = 'ARENA_WALL';
        description = 'Collision with outer perimeter boundary wall';
        confidence = 0.95;
      } else if (last.stance === 'COIL' || (prev && prev.stance === 'COIL')) {
        cause = 'ENCIRCLED_TRAP';
        description = 'Suffocated inside closed predator coil without open exit slit';
        confidence = 0.88;
      } else if (last.threatType === 'head' && last.threatDist <= headRadius * 2.8) {
        const isFacingUs = Math.abs(last.threatAngle || 0) <= Math.PI * 0.45;
        if (isFacingUs && (last.threatMassRatio || 1.0) > 1.1) {
          cause = 'HEAD_ON_RAM';
          description = 'Frontal collision with larger predator head';
          confidence = 0.92;
        } else {
          cause = 'HEAD_ON_JOUST';
          description = 'Simultaneous head-to-head collision';
          confidence = 0.85;
        }
      } else if (last.threatType === 'body' && last.threatDist <= headRadius * 3.2) {
        if (last.stance === 'ATTACK' || (prev && prev.stance === 'ATTACK')) {
          cause = 'ATTACK_OVEREXTENDED';
          description = 'Counter-intercepted while executing offensive strike';
          confidence = 0.86;
        } else if ((last.threatSpeed || 5.9) > 8.0) {
          cause = 'FLANK_CUTOFF';
          description = 'Cut off by high-speed enemy boost trajectory';
          confidence = 0.90;
        } else {
          cause = 'BODY_COLLISION';
          description = 'Impacted enemy body segments';
          confidence = 0.80;
        }
      } else if (last.isBoosting && meta.finalScore < 15 && meta.survivalTime > 10) {
        cause = 'BOOST_STARVATION';
        description = 'Starved mass down to zero fuel threshold during sprint';
        confidence = 0.82;
      } else if (last.threatDist <= headRadius * 4.0 && last.threatType !== 'self' && last.threatType !== 'none') {
        cause = 'BODY_COLLISION';
        description = 'Unavoidable collision with dense body segments';
        confidence = 0.75;
      }
    }

    const autopsy = {
      cause,
      description,
      confidence,
      fatalStance: last ? last.stance : 'UNKNOWN',
      survivalTime: Math.round((meta.survivalTime || 0) * 10) / 10,
      finalScore: Math.round(meta.finalScore || 0),
      peakLength: Math.round(meta.peakLength || 0),
      isHumanDriven: Boolean(meta.isHumanDriven),
      generation: meta.generation || 1,
      episode: meta.episode || 1,
      timestamp: Date.now(),
      killerProfile: (last && last.threatType && last.threatType !== 'none' && last.threatType !== 'self') ? {
        type: last.threatType,
        dist: Math.round((last.threatDist || 0) * 10) / 10,
        massRatio: Math.round((last.threatMassRatio || 1.0) * 100) / 100,
        speed: Math.round((last.threatSpeed || 5.9) * 10) / 10
      } : null,
      preCrashTelemetry: last ? {
        x: Math.round(last.x || 0),
        y: Math.round(last.y || 0),
        speed: Math.round((last.sp || 5.9) * 10) / 10,
        turnDelta: Math.round((last.turnDelta || 0) * 1000) / 1000,
        isBoosting: Boolean(last.isBoosting),
        dominantRayIdx: last.dominantRayIdx ?? -1,
        dominantRayDist: Math.round((last.dominantRayDist || 1.0) * 100) / 100
      } : null,
      incidentsCount: this.incidents.length
    };

    this.lastDeathAutopsy = autopsy;
    this.lastLifeWaypoints = this.waypoints.slice(-40);

    this.recentEpisodes.push({
      gen: autopsy.generation,
      ep: autopsy.episode,
      durationS: autopsy.survivalTime,
      peakLength: autopsy.peakLength,
      finalScore: autopsy.finalScore,
      deathCause: autopsy.cause,
      fatalStance: autopsy.fatalStance
    });
    if (this.recentEpisodes.length > this.maxRecentEpisodes) {
      this.recentEpisodes.shift();
    }

    return autopsy;
  }

  _getOrderedFrames() {
    if (this.frameBuffer.length < this.bufferCapacity) {
      return [...this.frameBuffer];
    }
    return [
      ...this.frameBuffer.slice(this.bufferIndex),
      ...this.frameBuffer.slice(0, this.bufferIndex)
    ];
  }

  _calculateStanceDutyCycles() {
    const total = Object.values(this.stanceCounts).reduce((a, b) => a + b, 0);
    if (total === 0) return {};
    const result = {};
    for (const [stance, count] of Object.entries(this.stanceCounts)) {
      if (count > 0) {
        result[stance] = `${((count / total) * 100).toFixed(1)}%`;
      }
    }
    return result;
  }

  _calculateZoneDistribution() {
    const total = this.zoneCounts.center + this.zoneCounts.midfield + this.zoneCounts.perimeter;
    if (total === 0) return { center: '0%', midfield: '0%', perimeter: '0%' };
    return {
      center: `${((this.zoneCounts.center / total) * 100).toFixed(1)}%`,
      midfield: `${((this.zoneCounts.midfield / total) * 100).toFixed(1)}%`,
      perimeter: `${((this.zoneCounts.perimeter / total) * 100).toFixed(1)}%`
    };
  }

  getTelemetrySummary() {
    const useLastLife = this.totalFrames < 30 && this.lastLifeFrames > 0;
    return {
      version: '1.0.0',
      totalRecordedFrames: useLastLife ? this.lastLifeFrames : this.totalFrames,
      lastDeathAutopsy: this.lastDeathAutopsy,
      recentEpisodeHistory: [...this.recentEpisodes],
      growthMilestones: useLastLife ? { ...this.lastLifeMilestones } : { ...this.growthMilestones },
      stanceDutyCycles: useLastLife ? { ...this.lastLifeDutyCycles } : this._calculateStanceDutyCycles(),
      zoneDistribution: useLastLife ? { ...this.lastLifeZones } : this._calculateZoneDistribution(),
      pathfinderBreadcrumbs: this.lastLifeWaypoints.length > 0
        ? [...this.lastLifeWaypoints]
        : this.waypoints.slice(-40)
    };
  }

  loadTelemetry(telemetry) {
    if (!telemetry || typeof telemetry !== 'object') return;
    if (typeof telemetry.totalRecordedFrames === 'number') {
      this.totalFrames = telemetry.totalRecordedFrames;
      this.lastLifeFrames = telemetry.totalRecordedFrames;
    }
    if (telemetry.lastDeathAutopsy) {
      this.lastDeathAutopsy = telemetry.lastDeathAutopsy;
    }
    if (Array.isArray(telemetry.recentEpisodeHistory)) {
      this.recentEpisodes = telemetry.recentEpisodeHistory.slice(-this.maxRecentEpisodes);
    }
    if (telemetry.growthMilestones && typeof telemetry.growthMilestones === 'object') {
      this.growthMilestones = { ...telemetry.growthMilestones };
      this.lastLifeMilestones = { ...telemetry.growthMilestones };
    }
    if (telemetry.stanceDutyCycles && typeof telemetry.stanceDutyCycles === 'object') {
      this.lastLifeDutyCycles = { ...telemetry.stanceDutyCycles };
    }
    if (telemetry.zoneDistribution && typeof telemetry.zoneDistribution === 'object') {
      this.lastLifeZones = { ...telemetry.zoneDistribution };
    }
    if (Array.isArray(telemetry.pathfinderBreadcrumbs)) {
      this.lastLifeWaypoints = telemetry.pathfinderBreadcrumbs.slice(-40);
    }
  }
}
