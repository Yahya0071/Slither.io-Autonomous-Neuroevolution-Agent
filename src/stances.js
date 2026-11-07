/**
 * Tactical Combat Stance State Machine, Steering Arbiter & Mass-ROI Boost Governor
 * 7 Tactical Stances: FORAGE, DEFEND, ATTACK, COIL, CONSTRICT, VACUUM, HUNT
 */

export const STANCES = ['FORAGE', 'DEFEND', 'ATTACK', 'COIL', 'CONSTRICT', 'VACUUM', 'HUNT'];

export function arbitrateStance({
  state,
  currentScore,
  scale,
  dangerUrgency,
  massRatio,
  normWallDist,
  wallDanger,
  cutOffThreat,
  attackCandidate,
  constrictCandidate,
  huntCandidate,
  outerPredatorNearby,
  richCorpseNearby,
  lastKillEpicenter,
  vacuumActiveUntil,
  now,
  inputs,
  headRadius,
  maxCorridorDist
}) {
  const canCoil = currentScore >= 45 || scale >= 1.05;
  const pinched = inputs[29] > 0.65 && inputs[30] > 0.65;
  const surrounded = dangerUrgency >= 0.55 && inputs[31] >= 0.33;
  const encirclementEscape = state.encirclementEscape;

  let stance = 'FORAGE';

  if (encirclementEscape && encirclementEscape.isOpen) {
    stance = 'DEFEND';
  } else if (encirclementEscape && encirclementEscape.isEarly && !encirclementEscape.isOpen) {
    stance = 'DEFEND';
  } else if (canCoil && (pinched || (surrounded && maxCorridorDist <= headRadius * 3.5))) {
    stance = 'COIL';
  } else if (cutOffThreat || dangerUrgency >= 0.5 || wallDanger) {
    stance = 'DEFEND';
  } else if (constrictCandidate && !outerPredatorNearby && massRatio < 1.15 && dangerUrgency < 0.25 && normWallDist < 0.88) {
    stance = 'CONSTRICT';
  } else if (((lastKillEpicenter && now < vacuumActiveUntil) || richCorpseNearby) && dangerUrgency < 0.38) {
    stance = 'VACUUM';
  } else if (attackCandidate && massRatio <= 1.5 && inputs[8] < 0.65 && dangerUrgency < 0.3) {
    stance = 'ATTACK';
  } else if (huntCandidate && dangerUrgency < 0.25 && !wallDanger && normWallDist < 0.86) {
    stance = 'HUNT';
  }

  return stance;
}

export function computeSteering({
  stance,
  state,
  agent,
  snake,
  currentAng,
  turnDelta,
  reflexSign,
  dangerUrgency,
  normWallDist,
  diffCenterAngle,
  headRadius,
  foodTarget,
  now,
  dt
}) {
  let steer = turnDelta;
  let activeStance = stance;
  let encirclementEscape = state.encirclementEscape;
  const maxCorridorDist = state.maxCorridorDist || 0;
  const bestCorridorAngle = state.bestCorridorAngle || currentAng;
  const encirclingThreatRays = state.encirclingThreatRays || 0;
  const selfShieldSide = state.selfShieldSide || 0;
  const cutOffThreat = state.cutOffThreat;
  const frontObstacleType = state.frontObstacleType;
  const inputs = state.inputs;
  const massRatio = agent.massRatio || 1.0;
  const wallDanger = normWallDist > 0.91;

  if (activeStance === 'COIL') {
    agent.foodSteerEma *= 0.2;
    if (maxCorridorDist > headRadius * 2.5 && encirclingThreatRays >= 4) {
      activeStance = 'DEFEND';
      agent.coilSign = 0;
      agent.combatStance = 'DEFEND';
      if (!encirclementEscape) {
        encirclementEscape = {
          threatRays: encirclingThreatRays,
          slitDist: maxCorridorDist,
          slitAngle: bestCorridorAngle,
          isOpen: true,
          isEarly: encirclingThreatRays < 7
        };
        state.encirclementEscape = encirclementEscape;
      }
    } else if (!agent.coilSign) {
      agent.coilSign = Math.abs(agent.prevTurnEma) > 0.25
        ? Math.sign(agent.prevTurnEma)
        : reflexSign;
    }
    if (activeStance === 'COIL') steer = agent.coilSign * 0.85;
  }

  if (activeStance !== 'COIL') {
    agent.coilSign = 0;
    if (activeStance === 'DEFEND') {
      agent.foodSteerEma *= 0.5;
      if (encirclementEscape && encirclementEscape.isOpen) {
        const breachTag = encirclementEscape.isEarly ? 'SLIT_BREACH_EARLY' : 'SLIT_BREACH_ESCAPE';
        agent.flightRecorder.logIncident(breachTag, {
          t: agent.survivalTime,
          x: snake.xx,
          y: snake.yy,
          info: `Rays: ${encirclementEscape.threatRays} | Slit angle: ${encirclementEscape.slitAngle.toFixed(2)} | Dist: ${encirclementEscape.slitDist.toFixed(0)}`
        });
        let slitError = encirclementEscape.slitAngle - currentAng;
        while (slitError > Math.PI) slitError -= Math.PI * 2;
        while (slitError < -Math.PI) slitError += Math.PI * 2;
        steer = Math.max(-1, Math.min(1, slitError / 1.25));
      } else if (encirclementEscape && encirclementEscape.isEarly) {
        agent.flightRecorder.logIncident('ENCIRCLEMENT_DETECTED', {
          t: agent.survivalTime,
          x: snake.xx,
          y: snake.yy,
          info: `Rays: ${encirclementEscape.threatRays} | Best clearance: ${encirclementEscape.slitDist.toFixed(0)}`
        });
        let slitError = encirclementEscape.slitAngle - currentAng;
        while (slitError > Math.PI) slitError -= Math.PI * 2;
        while (slitError < -Math.PI) slitError += Math.PI * 2;
        steer = Math.max(-1, Math.min(1, slitError / 1.65));
      } else if (wallDanger) {
        steer = diffCenterAngle > 0 ? 1.0 : -1.0;
      } else if (cutOffThreat) {
        if (selfShieldSide !== 0 && selfShieldSide === -cutOffThreat.cutSide) {
          agent.flightRecorder.logIncident('POCKET_SHIELD_DEFENSE', {
            t: agent.survivalTime,
            x: snake.xx,
            y: snake.yy,
            info: `Shield side: ${selfShieldSide}`
          });
          steer = selfShieldSide * 1.0;
        } else {
          steer = -cutOffThreat.cutSide * 1.0;
        }
      } else {
        if (frontObstacleType === 'head' && selfShieldSide !== 0) {
          steer = selfShieldSide * 1.0;
        } else if (inputs[31] >= 0.33 && maxCorridorDist > headRadius * 3.5) {
          let corridorError = bestCorridorAngle - currentAng;
          while (corridorError > Math.PI) corridorError -= Math.PI * 2;
          while (corridorError < -Math.PI) corridorError += Math.PI * 2;

          if (Math.abs(corridorError) > 2.2) {
            if (!agent.corridorTurnSign) {
              agent.corridorTurnSign = Math.abs(agent.prevTurnEma) > 0.15
                ? Math.sign(agent.prevTurnEma)
                : reflexSign;
            }
            steer = agent.corridorTurnSign * 1.0;
          } else {
            agent.corridorTurnSign = 0;
            steer = Math.max(-1, Math.min(1, corridorError / 1.35));
          }
        } else {
          const evasionTurn = (massRatio > 1.4) ? 1.0 : Math.max(0.9, Math.abs(steer));
          steer = reflexSign * evasionTurn;
        }
      }
    } else if (activeStance === 'ATTACK' && state.attackCandidate) {
      agent.foodSteerEma *= 0.5;
      const ac = state.attackCandidate;
      let attackAngle = Math.atan2(ac.interceptY - snake.yy, ac.interceptX - snake.xx);
      let attackError = attackAngle - currentAng;
      while (attackError > Math.PI) attackError -= Math.PI * 2;
      while (attackError < -Math.PI) attackError += Math.PI * 2;

      const closeInFactor = Math.max(0, Math.min(1, (headRadius * 8 - ac.distToIntercept) / (headRadius * 5)));
      const crossingCurl = (ac.side > 0 ? 0.22 : -0.22) * closeInFactor;
      steer = Math.max(-1, Math.min(1, (attackError / 1.35) + crossingCurl));
    } else if (activeStance === 'CONSTRICT' && state.constrictCandidate) {
      agent.foodSteerEma *= 0.2;
      const cc = state.constrictCandidate;
      const angleToEnemy = Math.atan2(cc.enemyY - snake.yy, cc.enemyX - snake.xx);
      const wrapSign = cc.side;

      const tightenFactor = Math.min(1.0, (agent.constrictDuration || 0) / 2.5);
      const inwardOffset = Math.PI * (0.45 + tightenFactor * 0.17);
      const orbitAngle = angleToEnemy + (wrapSign > 0 ? -inwardOffset : inwardOffset);

      let orbitError = orbitAngle - currentAng;
      while (orbitError > Math.PI) orbitError -= Math.PI * 2;
      while (orbitError < -Math.PI) orbitError += Math.PI * 2;
      steer = Math.max(-1, Math.min(1, orbitError / 1.25));
    } else if (activeStance === 'HUNT' && state.huntCandidate) {
      agent.foodSteerEma *= 0.3;
      const hc = state.huntCandidate;
      const angleToHunt = Math.atan2(hc.enemyY - snake.yy, hc.enemyX - snake.xx);
      let huntError = angleToHunt - currentAng;
      while (huntError > Math.PI) huntError -= Math.PI * 2;
      while (huntError < -Math.PI) huntError += Math.PI * 2;

      const huntAggression = hc.isSmallerPrey ? 0.85 : 0.55;
      const leadAngle = hc.enemyAng;
      let leadError = leadAngle - currentAng;
      while (leadError > Math.PI) leadError -= Math.PI * 2;
      while (leadError < -Math.PI) leadError += Math.PI * 2;
      const blendedError = huntError * 0.7 + leadError * 0.3;
      steer = Math.max(-1, Math.min(1, blendedError / 1.4)) * huntAggression + steer * (1 - huntAggression);

      if (dangerUrgency > 0.05) {
        steer = steer * (1 - dangerUrgency * 0.6) + reflexSign * dangerUrgency * 0.6;
      }
    } else {
      agent.foodTargetError = foodTarget ? foodTarget.error : null;
      if (foodTarget) {
        let foodSteer = Math.max(-1, Math.min(1, foodTarget.error / 1.85));

        if (foodTarget.contestedByRival && foodTarget.rivalBearingSign !== 0) {
          const crossOffset = -foodTarget.rivalBearingSign * 0.28;
          foodSteer = Math.max(-1, Math.min(1, (foodTarget.error / 1.85) + crossOffset));
        } else if (activeStance === 'VACUUM' && foodTarget.value >= 8 && foodTarget.dist > 120 && foodTarget.dist < 600) {
          const crossOffset = foodTarget.error >= 0 ? 0.20 : -0.20;
          foodSteer = Math.max(-1, Math.min(1, (foodTarget.error / 1.85) + crossOffset));
        }

        const safetyGating = dangerUrgency > 0.20 ? Math.max(0, 1 - (dangerUrgency - 0.20) / 0.30) : 1.0;
        const steerWeight = activeStance === 'VACUUM' ? 0.65 : 0.45;
        agent.foodSteerEma = agent.foodSteerEma * (1 - steerWeight) + foodSteer * steerWeight;
        const assist = (activeStance === 'VACUUM' ? 0.95 : 0.85) * (1 - dangerUrgency) * safetyGating;
        steer = steer * (1 - assist) + agent.foodSteerEma * assist;

        if (activeStance === 'VACUUM' && dangerUrgency > 0.15) {
          steer = steer * (1 - dangerUrgency * 0.7) + reflexSign * Math.min(1.0, dangerUrgency * 1.15);
        }
      } else {
        agent.foodSteerEma *= 0.8;
        if (dangerUrgency < 0.1 && normWallDist < 0.82) {
          steer += Math.sin(now * 0.003) * 0.08;
        }
      }

      if (dangerUrgency > 0.05) {
        steer = steer * (1 - dangerUrgency) + reflexSign * dangerUrgency;
      } else if (normWallDist > 0.86) {
        steer = Math.sign(diffCenterAngle) * Math.min(1.0, Math.abs(diffCenterAngle) + 0.35);
      } else if (normWallDist < 0.18 && inputs[31] >= 0.15 && dangerUrgency < 0.10) {
        const awayAngle = diffCenterAngle > 0 ? diffCenterAngle - Math.PI : diffCenterAngle + Math.PI;
        steer += Math.sign(awayAngle) * 0.10;
      }
    }
  }

  return { turnDelta: steer, activeStance };
}

export function governBoost({
  stance,
  state,
  agent,
  snake,
  turnDelta,
  currentAng,
  currentScore,
  currentSpeed,
  dangerUrgency,
  normWallDist,
  headRadius,
  foodTarget,
  boostProb
}) {
  const inputs = state.inputs;
  const forwardClear = inputs[7] < 0.35 && inputs[8] < 0.35 && inputs[9] < 0.35;
  const healthyLength = currentScore >= agent.minBoostScore &&
    !(agent.peakLength > 120 && currentScore < agent.peakLength * 0.9);
  const calmTraffic = inputs[31] < 0.5;
  const wallDanger = normWallDist > 0.91;
  const cutOffThreat = state.cutOffThreat;
  const massRatio = agent.massRatio || 1.0;
  const encirclementEscape = state.encirclementEscape;
  const maxCorridorDist = state.maxCorridorDist || 0;
  const bestCorridorAngle = state.bestCorridorAngle || currentAng;
  const attackCandidate = state.attackCandidate;
  const huntCandidate = state.huntCandidate;
  const slipstreamTarget = state.slipstreamTarget;

  let shouldBoost = false;

  if (stance === 'COIL') {
    shouldBoost = false;
  } else if (stance === 'DEFEND') {
    if (encirclementEscape && encirclementEscape.isOpen) {
      let slitError = encirclementEscape.slitAngle - currentAng;
      while (slitError > Math.PI) slitError -= Math.PI * 2;
      while (slitError < -Math.PI) slitError += Math.PI * 2;
      shouldBoost = Math.abs(slitError) < 0.75 && agent.boostFuel > 0.10 && agent.boostCooldown <= 0;
    } else if (encirclementEscape && encirclementEscape.isEarly) {
      let slitError = encirclementEscape.slitAngle - currentAng;
      while (slitError > Math.PI) slitError -= Math.PI * 2;
      while (slitError < -Math.PI) slitError += Math.PI * 2;
      shouldBoost = Math.abs(slitError) < 0.55 && agent.boostFuel > 0.20 && agent.boostCooldown <= 0;
    } else if (wallDanger || (cutOffThreat && (cutOffThreat.threatScore > 0.65 || cutOffThreat.isBoosting)) || (massRatio > 1.4 && dangerUrgency >= 0.35)) {
      shouldBoost = false;
    } else if (forwardClear && state.forwardObstacleDanger < 0.20 && state.minObstacleDist > headRadius * 5.0 &&
               healthyLength && agent.boostFuel > 0.25 && agent.boostCooldown <= 0) {
      shouldBoost = true;
    } else if (inputs[31] >= 0.33 && maxCorridorDist > headRadius * 3.5 &&
               agent.boostFuel > 0.15 && agent.boostCooldown <= 0) {
      let cDiff = bestCorridorAngle - currentAng;
      while (cDiff > Math.PI) cDiff -= Math.PI * 2;
      while (cDiff < -Math.PI) cDiff += Math.PI * 2;
      shouldBoost = Math.abs(cDiff) < 0.65;
    } else {
      shouldBoost = false;
    }
  } else if (stance === 'ATTACK') {
    shouldBoost = Boolean(attackCandidate && attackCandidate.canBoostAttack) && inputs[8] < 0.65 && agent.boostCooldown <= 0 && agent.boostFuel > 0.20;
  } else if (stance === 'CONSTRICT') {
    shouldBoost = false;
  } else if (stance === 'VACUUM') {
    shouldBoost = forwardClear && healthyLength && Boolean(foodTarget) && foodTarget.dist < 400 &&
                  Math.abs(foodTarget.error) < 0.50 &&
                  !foodTarget.contestedByRival && inputs[31] < 0.5 && dangerUrgency < 0.15 &&
                  agent.boostFuel > 0.20 && agent.boostCooldown <= 0;
  } else if (stance === 'HUNT') {
    const isSmallPrey = huntCandidate && huntCandidate.isSmallerPrey;
    const huntAngle = huntCandidate ? Math.atan2(huntCandidate.enemyY - snake.yy, huntCandidate.enemyX - snake.xx) : currentAng;
    let huntAlignError = huntAngle - currentAng;
    while (huntAlignError > Math.PI) huntAlignError -= Math.PI * 2;
    while (huntAlignError < -Math.PI) huntAlignError += Math.PI * 2;
    const wellAligned = Math.abs(huntAlignError) < 0.55;
    shouldBoost = isSmallPrey && wellAligned && forwardClear && healthyLength &&
                  agent.boostFuel > 0.30 && agent.boostCooldown <= 0;
  } else {
    const estBurnTimeSec = foodTarget ? (foodTarget.dist / (currentSpeed * 1.85 * 60)) : 1.0;
    const estMassBurned = estBurnTimeSec * 16.0;
    const positiveROI = Boolean(foodTarget) && (foodTarget.value * 2.5 >= estMassBurned);

    const worthSprinting = Boolean(foodTarget) &&
      foodTarget.value >= 8 &&
      foodTarget.dist < 450 &&
      Math.abs(foodTarget.error) < 0.6 &&
      positiveROI;

    const isPreyChase = Boolean(foodTarget && foodTarget.isPrey &&
      foodTarget.dist < 700 && Math.abs(foodTarget.error) < 0.70);

    const isSlipstreamSprint = Boolean(slipstreamTarget && foodTarget && foodTarget.dist < 420 && Math.abs(foodTarget.error) < 0.55);

    const maySprint = forwardClear && healthyLength && calmTraffic && (worthSprinting || isPreyChase || isSlipstreamSprint) &&
      agent.boostCooldown <= 0 && agent.boostFuel > 0.15;

    const policyVeto = !isPreyChase && !isSlipstreamSprint && boostProb < 0.15;
    shouldBoost = maySprint && !policyVeto;
  }

  if (shouldBoost && stance !== 'DEFEND' && Math.abs(turnDelta) > 0.48) {
    shouldBoost = false;
  }

  return shouldBoost;
}
