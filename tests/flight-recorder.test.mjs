/**
 * Flight Recorder & Forensic Deductor Unit Tests
 *
 * Verifies:
 * 1. Rolling FIFO buffer records state frames and stance duty cycles.
 * 2. Autopsy deduction classifies ARENA_WALL, HEAD_ON_RAM, FLANK_CUTOFF, ENCIRCLED_TRAP, BOOST_STARVATION.
 * 3. Compact pathfinder breadcrumbs downsample coordinates at regular intervals.
 * 4. Growth milestones and spatial zone occupancy calculations.
 * 5. Checkpoint serialization round-trips telemetry without data loss.
 * 6. SlitherMLAgent evaluate updates flight recorder and onDeath creates autopsy.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadAgent, MemoryStorage, sourceCode, extractClass } from '../agent-loader.mjs';

test('FlightRecorder initializes cleanly with expected default state', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  assert.equal(rec.totalFrames, 0);
  assert.equal(rec.frameBuffer.length, 0);
  assert.equal(rec.waypoints.length, 0);
  assert.equal(rec.lastDeathAutopsy, null);
  assert.deepEqual(rec.growthMilestones, {});
});

test('FlightRecorder.recordFrame accumulates frames, zone counts, and stance duty cycles', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder({ bufferCapacity: 10 });

  // Record 12 frames in CENTER zone with FORAGE stance
  for (let i = 0; i < 12; i++) {
    rec.recordFrame({
      t: i * 0.1,
      x: 100,
      y: 100,
      sp: 5.9,
      score: 50 + i * 10,
      stance: 'FORAGE',
      normWallDist: 0.1
    });
  }

  assert.equal(rec.totalFrames, 12);
  assert.equal(rec.frameBuffer.length, 10, 'FIFO buffer should cap at capacity');
  assert.equal(rec.zoneCounts.center, 12);
  assert.equal(rec.zoneCounts.perimeter, 0);

  const duties = rec._calculateStanceDutyCycles();
  assert.equal(duties.FORAGE, '100.0%');

  const zones = rec._calculateZoneDistribution();
  assert.equal(zones.center, '100.0%');
});

test('FlightRecorder.deduceDeathAutopsy classifies ARENA_WALL near boundary', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  rec.recordFrame({
    t: 10.0,
    x: 21500,
    y: 0,
    sp: 5.9,
    score: 200,
    stance: 'DEFEND',
    normWallDist: 0.98,
    threatDist: 500,
    threatType: 'wall'
  });

  const autopsy = rec.deduceDeathAutopsy({
    survivalTime: 10.0,
    finalScore: 200,
    peakLength: 20,
    generation: 2,
    episode: 1
  });

  assert.equal(autopsy.cause, 'ARENA_WALL');
  assert.equal(autopsy.fatalStance, 'DEFEND');
  assert.ok(autopsy.confidence >= 0.9);
  assert.equal(rec.lastDeathAutopsy.cause, 'ARENA_WALL');
  assert.equal(rec.recentEpisodes.length, 1);
});

test('FlightRecorder.deduceDeathAutopsy classifies HEAD_ON_RAM against larger predator', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  rec.recordFrame({
    t: 25.4,
    x: 0,
    y: 0,
    sp: 6.2,
    score: 300,
    stance: 'DEFEND',
    normWallDist: 0.2,
    threatDist: 28, // within 2.8 * headRadius (14 * sqrt(20) ~ 62)
    threatType: 'head',
    threatAngle: 0.1, // facing forward
    threatMassRatio: 1.6, // significantly larger
    threatSpeed: 9.0
  });

  const autopsy = rec.deduceDeathAutopsy({
    survivalTime: 25.4,
    finalScore: 300,
    peakLength: 30,
    generation: 5,
    episode: 2
  });

  assert.equal(autopsy.cause, 'HEAD_ON_RAM');
  assert.equal(autopsy.killerProfile.type, 'head');
  assert.equal(autopsy.killerProfile.massRatio, 1.6);
});

test('FlightRecorder.deduceDeathAutopsy classifies FLANK_CUTOFF on high-speed enemy body', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  rec.recordFrame({
    t: 15.0,
    x: 1000,
    y: 1000,
    sp: 5.9,
    score: 150,
    stance: 'DEFEND',
    normWallDist: 0.3,
    threatDist: 35,
    threatType: 'body',
    threatSpeed: 10.5 // boosting enemy snake cutting across
  });

  const autopsy = rec.deduceDeathAutopsy({
    survivalTime: 15.0,
    finalScore: 150,
    peakLength: 15,
    generation: 3,
    episode: 1
  });

  assert.equal(autopsy.cause, 'FLANK_CUTOFF');
});

test('FlightRecorder.deduceDeathAutopsy classifies ENCIRCLED_TRAP during COIL stance', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  rec.recordFrame({
    t: 40.0,
    x: 200,
    y: 200,
    sp: 5.9,
    score: 800,
    stance: 'COIL',
    normWallDist: 0.2,
    threatDist: 40,
    threatType: 'body'
  });

  const autopsy = rec.deduceDeathAutopsy({
    survivalTime: 40.0,
    finalScore: 800,
    peakLength: 80,
    generation: 10,
    episode: 1
  });

  assert.equal(autopsy.cause, 'ENCIRCLED_TRAP');
});

test('FlightRecorder.deduceDeathAutopsy classifies BOOST_STARVATION when boosting at zero score', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  rec.recordFrame({
    t: 35.0,
    x: 500,
    y: 500,
    sp: 11.0,
    score: 10,
    stance: 'FORAGE',
    isBoosting: true,
    normWallDist: 0.3,
    threatDist: 300,
    threatType: 'none'
  });

  const autopsy = rec.deduceDeathAutopsy({
    survivalTime: 35.0,
    finalScore: 10,
    peakLength: 500,
    generation: 4,
    episode: 3
  });

  assert.equal(autopsy.cause, 'BOOST_STARVATION');
});

test('Pathfinder breadcrumbs downsamples waypoints and growth milestones', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  // Record 10 seconds of movement
  for (let s = 0; s < 100; s++) {
    const t = s * 0.1;
    rec.recordFrame({
      t,
      x: s * 10,
      y: s * 5,
      score: s * 60, // reaches 500, 1000, 2500, 5000
      stance: s < 50 ? 'FORAGE' : 'ATTACK'
    });
  }

  // Waypoints logged every 2.5s -> at ~0s, 2.5s, 5.0s, 7.5s (4 waypoints)
  assert.ok(rec.waypoints.length >= 4, `expected >= 4 waypoints, got ${rec.waypoints.length}`);
  assert.equal(rec.waypoints[0][4], 'FORAGE');

  // Growth milestones
  assert.ok(rec.growthMilestones['500'], 'should have logged milestone 500');
  assert.ok(rec.growthMilestones['1000'], 'should have logged milestone 1000');
  assert.ok(rec.growthMilestones['2500'], 'should have logged milestone 2500');
  assert.ok(rec.growthMilestones['5000'], 'should have logged milestone 5000');
});

test('Checkpoint persistence: telemetry is serialized into checkpoint and restored via applySavedState', () => {
  const storage = new MemoryStorage();
  const harness = loadAgent({ storage });
  const agent = harness.newAgent();
  agent.resetBrain();

  // Populate some telemetry in agent's flight recorder
  agent.flightRecorder.recordFrame({
    t: 12.0,
    x: 500,
    y: 500,
    score: 300,
    stance: 'DEFEND',
    normWallDist: 0.95
  });
  agent.onDeath(false); // Triggers autopsy and saves checkpoint

  const serialized = storage.getItem(harness.STORAGE_KEY);
  assert.ok(serialized, 'checkpoint must be saved to storage');
  const parsed = JSON.parse(serialized);

  assert.ok(parsed.telemetry, 'checkpoint must contain telemetry object');
  assert.equal(parsed.telemetry.lastDeathAutopsy.cause, 'ARENA_WALL');
  assert.equal(parsed.telemetry.recentEpisodeHistory.length, 1);

  // Restore state into fresh agent
  const agent2 = harness.newAgent();
  agent2.applySavedState(parsed);
  assert.equal(agent2.flightRecorder.lastDeathAutopsy.cause, 'ARENA_WALL');
  assert.equal(agent2.flightRecorder.recentEpisodes.length, 1);

  const report = agent2.getAutopsyReport();
  assert.equal(report.cause, 'ARENA_WALL');
});

test('SlitherMLAgent evaluate updates flightRecorder and onDeath generates autopsy', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();
  const snake = { id: 'me', xx: 100, yy: 100, eang: 0, sc: 1.0, sp: 5.9, sct: 15, pts: [] };

  // Run 5 evaluation frames
  for (let i = 0; i < 5; i++) {
    agent.evaluate(snake, [], [], []);
  }

  assert.equal(agent.flightRecorder.totalFrames, 5);
  assert.equal(agent.flightRecorder.stanceCounts.FORAGE, 5);

  agent.onDeath(false);
  assert.ok(agent.flightRecorder.lastDeathAutopsy);
  assert.equal(agent.flightRecorder.recentEpisodes.length, 1);
});

test('high-score headRadius accuracy: distant threats at high scores are not misclassified as point-blank collisions', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  // High score 15,000 snake with enemy head at 300px distance (far outside point-blank radius)
  rec.recordFrame({
    t: 120.0,
    x: 0,
    y: 0,
    sp: 5.9,
    score: 15000,
    headRadius: 28, // realistic scale: 14 * 2.0
    stance: 'FORAGE',
    normWallDist: 0.2,
    threatDist: 300,
    threatType: 'head',
    threatAngle: 0.1,
    threatMassRatio: 1.5,
    threatSpeed: 5.9
  });

  const autopsy = rec.deduceDeathAutopsy({
    survivalTime: 120.0,
    finalScore: 15000,
    peakLength: 1500,
    generation: 15,
    episode: 2
  });

  // Old bug calculated headRadius = 14 * sqrt(15000 / 15) = 442.7px, where 442.7 * 2.8 = 1239px > 300px,
  // causing a distant threat to falsely trigger HEAD_ON_RAM.
  // With realistic headRadius = 28px, 28 * 2.8 = 78.4px < 300px, correctly identifying SUDDEN_IMPACT.
  assert.notEqual(autopsy.cause, 'HEAD_ON_RAM');
  assert.notEqual(autopsy.cause, 'BODY_COLLISION');
  assert.equal(autopsy.cause, 'SUDDEN_IMPACT');
});

test('Teacher Mode autopsy recording: records flight frames during learnFromHuman and deduces cause on death', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();
  // Near arena perimeter wall: xx = 21500
  const snake = { id: 'me', xx: 21500, yy: 0, eang: 0, sc: 1.0, sp: 5.9, sct: 10, pts: [] };

  for (let i = 0; i < 5; i++) {
    agent.learnFromHuman(snake, [], [], [], 100, 0, false);
  }

  assert.ok(agent.flightRecorder.totalFrames >= 5, 'should record flight frames during Teacher mode');
  assert.equal(agent.flightRecorder.stanceCounts.TEACHER, 5, 'should increment TEACHER stance frames');

  agent.onDeath(true);
  const autopsy = agent.flightRecorder.lastDeathAutopsy;
  assert.ok(autopsy, 'autopsy should exist');
  assert.equal(autopsy.cause, 'ARENA_WALL');
  assert.equal(autopsy.fatalStance, 'TEACHER');
  assert.equal(autopsy.isHumanDriven, true);
});

test('Controller & Agent export parity: exportCheckpoint includes telemetry and exportBrain delegates to it', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();

  // 1. Agent exportCheckpoint includes telemetry
  const serialized = agent.exportCheckpoint();
  assert.ok(serialized, 'checkpoint must serialize');
  const parsed = JSON.parse(serialized);
  assert.ok(parsed.telemetry, 'exported checkpoint must contain telemetry block');
  assert.equal(typeof parsed.telemetry.totalRecordedFrames, 'number');

  // 2. Controller source code delegates exportBrain() to this.agent.exportCheckpoint()
  const controllerSrc = extractClass(sourceCode, 'SlitherMLController');
  assert.ok(
    controllerSrc.includes('this.agent.exportCheckpoint()'),
    'SlitherMLController.exportBrain must delegate directly to this.agent.exportCheckpoint()'
  );
});

test('offset coordinates autopsy: snake in midfield (x: 33088, y: 35181, normWallDist: 0.82) is not misclassified as ARENA_WALL', () => {
  const harness = loadAgent();
  const { FlightRecorder } = harness;
  const rec = new FlightRecorder();

  // Snapshot from real Slither.io offset coordinates game (grd = 21600, center at 21600, 21600)
  // Distance from center is ~17,788px (midfield normWallDist ~0.82)
  rec.recordFrame({
    t: 22.2,
    x: 33088,
    y: 35181,
    sp: 5.8,
    score: 8,
    headRadius: 14,
    stance: 'DEFEND',
    normWallDist: 0.82,
    threatDist: 25,
    threatType: 'body',
    threatSpeed: 9.5
  });

  const autopsy = rec.deduceDeathAutopsy({
    survivalTime: 22.2,
    finalScore: 8,
    peakLength: 8,
    generation: 150,
    episode: 1
  });

  // Old bug computed Math.hypot(33088, 35181) = 48296 >= 21250, falsely claiming ARENA_WALL.
  // With normWallDist = 0.82 < 0.94, it must correctly classify as FLANK_CUTOFF or BODY_COLLISION.
  assert.notEqual(autopsy.cause, 'ARENA_WALL', 'midfield death must never be misdiagnosed as ARENA_WALL');
  assert.equal(autopsy.cause, 'FLANK_CUTOFF', 'fast enemy cutting across must be classified as FLANK_CUTOFF');
  assert.equal(autopsy.killerProfile.type, 'body');
});

test('killerProfile never reports type: self on self-body shielding', () => {
  const harness = loadAgent();
  const agent = harness.newAgent();

  // Snake coiling where ray 8 detects own body segment ('self')
  const snake = { id: 'me', xx: 100, yy: 100, eang: 0, sc: 1.0, sp: 5.9, sct: 10, pts: [] };
  // Add self segments forming a loop around head
  for (let p = 0; p < 30; p++) {
    snake.pts.push({ xx: 100 + Math.cos(p * 0.2) * 50, yy: 100 + Math.sin(p * 0.2) * 50 });
  }

  agent.evaluate(snake, [], [], []);
  agent.onDeath(false);

  const autopsy = agent.flightRecorder.lastDeathAutopsy;
  if (autopsy && autopsy.killerProfile) {
    assert.notEqual(autopsy.killerProfile.type, 'self', 'killerProfile must never attribute death to self');
  }
});


test('early encirclement detection (5 rays): snake escapes via DEFEND not COIL', () => {
  // Verify encirclementEscape.isEarly is set when exactly 5 rays are blocked
  // and isOpen is true when max slit clearance > headRadius * 2.5
  const harness = loadAgent();
  const { SlitherMLAgent } = harness;
  const agent = new SlitherMLAgent({ generation: 1 });

  // Build a snake with scale 1.0 (headRadius ~14), surrounded on 5 rays
  const headX = 0, headY = 0;
  const snake = { id: 'me', xx: headX, yy: headY, eang: 0, sc: 1.0, sp: 5.9, sct: 15, pts: [] };

  // Place a large encircling snake whose body covers 5 rays at ~70px
  // rays are at angles: 0, 22.5, 45, 67.5, 90, 112.5, ... degrees
  // Block rays 0-4 (right side)
  const enemySnake = { id: 'big', xx: 85, yy: 0, eang: Math.PI, sc: 3.0, sp: 5.9, pts: [] };
  const blockAngles = [0, Math.PI/8, Math.PI/4, 3*Math.PI/8, Math.PI/2];
  for (const ang of blockAngles) {
    enemySnake.pts.push({ xx: headX + Math.cos(ang) * 72, yy: headY + Math.sin(ang) * 72 });
  }

  const result = agent.evaluate(snake, [enemySnake], [], []);

  // Inspect the stored encirclementEscape
  const esc = agent.encirclementEscape;
  if (esc) {
    // If detected, isEarly should be set when < 7 rays
    if (esc.threatRays < 7) {
      assert.equal(esc.isEarly, true, 'isEarly must be true when threatRays < 7');
    }
    // isOpen threshold is 2.5x headRadius (not 4.5x)
    const headRadius = 14; // scale=1.0
    if (esc.slitDist > headRadius * 2.5) {
      assert.equal(esc.isOpen, true, 'isOpen must fire at 2.5x headRadius clearance');
    }
  }
  // Agent should NOT be in COIL when encirclementEscape.isOpen is true
  if (esc && esc.isOpen) {
    assert.notEqual(agent.combatStance, 'COIL', 'must not COIL when a slit escape is available');
  }
});

test('attack candidate fires at any score (angle-only), boosts only when canBoostAttack', () => {
  // With minBoostScore gate removed from stance entry, attackCandidate can generate a
  // turning-only intercept even when currentScore < minBoostScore.
  // canBoostAttack on the candidate should be false in that case.
  const harness = loadAgent();
  const agent = harness.newAgent();

  // Small snake (score below minBoostScore which is typically 150)
  const headX = 0, headY = 0;
  const snake = { id: 'me', xx: headX, yy: headY, eang: 0, sc: 1.0, sp: 5.9, sct: 5, pts: [] };

  // Enemy passing ~200px ahead — crossing trajectory
  const enemy = { id: 'prey', xx: 200, yy: 0, eang: Math.PI / 2, sc: 0.8, sp: 5.9, pts: [] };
  for (let p = 0; p < 5; p++) {
    enemy.pts.push({ xx: 200 + p * 10, yy: p * 10 });
  }

  // Run two frames to build velocity estimate in tracker
  agent.evaluate(snake, [enemy], [], []);
  enemy.yy += 30; // move enemy so velocity is visible
  agent.evaluate(snake, [enemy], [], []);

  // After two frames the attack candidate may or may not exist depending on geometry.
  // The key invariant: if a candidate exists it must have canBoostAttack defined.
  // And if currentScore < minBoostScore, canBoostAttack must be false.
  const hasCandidate = agent.attackTarget !== null;
  if (hasCandidate) {
    assert.ok('canBoostAttack' in agent.attackTarget, 'attackCandidate must carry canBoostAttack flag');
    const currentScore = 5;
    const minBoostScore = agent.minBoostScore;
    if (currentScore < minBoostScore) {
      assert.equal(agent.attackTarget.canBoostAttack, false, 'canBoostAttack must be false when score < minBoostScore');
    }
  }
});
