/**
 * Perception Engine: 16 Proximity Rays, 8 Food Sectors, Semantic Obstacle Classification,
 * and Combat Geometry Evaluation.
 */

export function extractCombatGeometry(snake, snakes, dt, now, enemyTracker, minBoostScore = 80, boostFuel = 1.0, currentScore = 10) {
  const headX = snake.xx;
  const headY = snake.yy;
  const currentAng = (typeof snake.eang === 'number') ? snake.eang : (snake.ang || 0);
  const scale = snake.sc || 1.0;
  const mySpeed = (typeof snake.sp === 'number' && snake.sp > 0) ? snake.sp : 5.9;
  const cosHead = Math.cos(currentAng);
  const sinHead = Math.sin(currentAng);

  let nearbyEnemyCount = 0;
  let cutOffThreat = null;
  let attackCandidate = null;
  let bestAttackScore = -1;
  let constrictCandidate = null;
  let bestConstrictScore = -1;
  let huntCandidate = null;
  let nearestThreatDist = Infinity;
  let nearestThreatScale = scale;
  let nearestThreatSpeed = 5.9;
  let outerPredatorNearby = false;
  let slipstreamTarget = null;

  const activeIds = new Set();
  if (Array.isArray(snakes)) {
    for (let i = 0; i < snakes.length; i++) {
      const other = snakes[i];
      if (!other || other.dead || other === snake || other.id === snake.id) continue;

      const dxHead = other.xx - headX;
      const dyHead = other.yy - headY;
      const distHead = Math.hypot(dxHead, dyHead);
      const otherScale = other.sc || 1.0;
      const otherSpeed = (typeof other.sp === 'number' && other.sp > 0) ? other.sp : 5.9;

      if (distHead < 520) {
        nearbyEnemyCount++;
      }
      if (distHead < nearestThreatDist) {
        nearestThreatDist = distHead;
        nearestThreatScale = otherScale;
        nearestThreatSpeed = otherSpeed;
      }
      if (distHead < 550 && otherScale >= scale * 1.15) {
        outerPredatorNearby = true;
      }

      if (otherSpeed > 7.0 && distHead > 70 && distHead < 650) {
        let relBearing = Math.atan2(dyHead, dxHead) - currentAng;
        while (relBearing > Math.PI) relBearing -= Math.PI * 2;
        while (relBearing < -Math.PI) relBearing += Math.PI * 2;
        if (Math.abs(relBearing) < Math.PI * 0.35) {
          slipstreamTarget = {
            enemyId: other.id !== undefined ? other.id : other,
            dist: distHead,
            speed: otherSpeed,
            side: relBearing >= 0 ? 1 : -1
          };
        }
      }

      const id = other.id !== undefined ? other.id : other;
      activeIds.add(id);
      const prev = enemyTracker.get(id);
      let vx = 0;
      let vy = 0;
      let omega = 0;
      const otherAng = (typeof other.eang === 'number') ? other.eang : (other.ang || 0);

      if (prev) {
        const dtTrack = (now - prev.timestamp) / 1000;
        if (dtTrack > 0.005) {
          const instVx = (other.xx - prev.xx) / dtTrack;
          const instVy = (other.yy - prev.yy) / dtTrack;
          vx = prev.vx * 0.6 + instVx * 0.4;
          vy = prev.vy * 0.6 + instVy * 0.4;
          if (typeof prev.ang === 'number') {
            let dAng = otherAng - prev.ang;
            while (dAng > Math.PI) dAng -= Math.PI * 2;
            while (dAng < -Math.PI) dAng += Math.PI * 2;
            omega = (prev.omega || 0) * 0.6 + (dAng / dtTrack) * 0.4;
          }
          enemyTracker.set(id, { xx: other.xx, yy: other.yy, ang: otherAng, vx, vy, omega, timestamp: now, wasClose: distHead < 300, dist: distHead, enemyScale: otherScale });
        } else {
          vx = prev.vx;
          vy = prev.vy;
          omega = prev.omega || 0;
        }
      } else {
        enemyTracker.set(id, { xx: other.xx, yy: other.yy, ang: otherAng, vx: 0, vy: 0, omega: 0, timestamp: now, wasClose: distHead < 300, dist: distHead, enemyScale: otherScale });
      }

      if (distHead < 550) {
        const enemySpeedSec = Math.hypot(vx, vy) || (otherSpeed * 60);
        const evx = vx !== 0 ? vx : Math.cos(otherAng) * enemySpeedSec;
        const evy = vy !== 0 ? vy : Math.sin(otherAng) * enemySpeedSec;

        let relBearing = Math.atan2(dyHead, dxHead) - currentAng;
        while (relBearing > Math.PI) relBearing -= Math.PI * 2;
        while (relBearing < -Math.PI) relBearing += Math.PI * 2;

        if (Math.abs(relBearing) < Math.PI * 0.75) {
          const D = cosHead * evy - sinHead * evx;
          if (Math.abs(D) > 1e-4) {
            const t_self = (dxHead * evy - dyHead * evx) / D;
            const t_enemy = (dxHead * sinHead - dyHead * cosHead) / D;
            const myReachTime = t_self / (mySpeed * 60);
            const isEnemyBoosting = otherSpeed > 7.0;
            const maxReachDist = isEnemyBoosting ? 500 : 420;
            const reachTimeTolerance = isEnemyBoosting ? 0.55 : 0.35;

            if (t_enemy > 0 && t_enemy < 1.25 && t_self > 8 && t_self < maxReachDist && t_enemy <= myReachTime + reachTimeTolerance) {
              const cutSide = relBearing >= 0 ? 1 : -1;
              const relScale = otherScale / scale;
              const scaleWeight = Math.max(0.5, Math.min(3.0, relScale));
              const threatScore = (1 - distHead / maxReachDist) * (isEnemyBoosting ? 1.6 : 1.0) * scaleWeight;
              if (!cutOffThreat || threatScore > cutOffThreat.threatScore) {
                cutOffThreat = { enemyId: id, cutSide, dist: distHead, threatScore, massRatio: relScale, enemyScale: otherScale, isBoosting: isEnemyBoosting };
              }
            }
          }
        }

        if (distHead >= 60 && distHead <= 500) {
          const enemyScale = otherScale;
          if (enemyScale <= scale * 1.6) {
            const relSpeed = Math.hypot(evx - (cosHead * mySpeed * 60), evy - (sinHead * mySpeed * 60));
            const leadSec = Math.max(0.25, Math.min(0.65, distHead / (relSpeed + 1e-3)));

            let interceptX, interceptY;
            if (Math.abs(omega) > 0.20 && Math.abs(omega) < 6.0) {
              const enemySpeed = Math.hypot(evx, evy);
              const turnRad = enemySpeed / omega;
              interceptX = other.xx + (turnRad * (Math.sin(otherAng + omega * leadSec) - Math.sin(otherAng)));
              interceptY = other.yy - (turnRad * (Math.cos(otherAng + omega * leadSec) - Math.cos(otherAng)));
            } else {
              interceptX = other.xx + evx * leadSec;
              interceptY = other.yy + evy * leadSec;
            }

            const distToIntercept = Math.hypot(interceptX - headX, interceptY - headY);
            const sprintSpeedSec = mySpeed * 1.85 * 60;
            const timeToIntercept = distToIntercept / sprintSpeedSec;

            if (timeToIntercept <= leadSec + 0.35 && distToIntercept < 560) {
              const canBoostAttack = currentScore >= minBoostScore && boostFuel > 0.20;
              const attackScore = ((scale / enemyScale) * 120) / (distToIntercept + 40);
              if (attackScore > bestAttackScore) {
                bestAttackScore = attackScore;
                attackCandidate = {
                  enemyId: id,
                  interceptX,
                  interceptY,
                  distToIntercept,
                  side: relBearing >= 0 ? 1 : -1,
                  score: attackScore,
                  canBoostAttack
                };
              }
            }
          }
        }

        if (!outerPredatorNearby && (currentScore >= 500 || scale >= 1.25) && distHead >= 70 && distHead <= 480 && Math.abs(relBearing) >= 0.65) {
          const enemyScale = otherScale;
          const enemyScore = (typeof other.sct === 'number' && other.sct > 0) ? other.sct : Math.floor(15 * (enemyScale - 1) / 0.001 || 10);
          if (enemyScale <= scale * 0.65 && enemyScore <= currentScore * 0.45) {
            const constrictScore = ((scale / enemyScale) * 100) / (distHead + 40);
            if (constrictScore > bestConstrictScore) {
              bestConstrictScore = constrictScore;
              constrictCandidate = {
                enemyId: id,
                enemyX: other.xx,
                enemyY: other.yy,
                enemyScale,
                dist: distHead,
                side: relBearing >= 0 ? 1 : -1
              };
            }
          }
        }

        if (!huntCandidate || distHead < huntCandidate.dist) {
          const enemyScale = otherScale;
          if (enemyScale <= scale * 2.0 && distHead < 700) {
            huntCandidate = {
              enemyId: id,
              enemyX: other.xx,
              enemyY: other.yy,
              enemyScale,
              dist: distHead,
              enemyAng: otherAng,
              isSmallerPrey: enemyScale < scale * 0.85
            };
          }
        }
      }
    }
  }

  return {
    activeIds,
    nearbyEnemyCount,
    cutOffThreat,
    attackCandidate,
    constrictCandidate,
    huntCandidate,
    nearestThreatDist,
    nearestThreatScale,
    nearestThreatSpeed,
    outerPredatorNearby,
    slipstreamTarget
  };
}

export function castProximityRays({
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
  rayCount = 16,
  enemyTracker
}) {
  const activeRays = [];
  const inputs = new Float32Array(rayCount);

  let minObstacleDist = sensorRange;
  let frontObstacleDist = sensorRange;
  let frontThreatScale = scale;
  let frontThreatSpeed = 5.9;
  let closestObstacleType = 'none';
  let frontObstacleType = 'none';
  let closestObstacleAngle = 0;
  let frontObstacleAngle = 0;

  let minLethalDist = sensorRange;
  let closestLethalType = 'none';
  let closestLethalScale = scale;
  let closestLethalSpeed = 5.9;
  let closestLethalAngle = 0;
  let frontLethalDist = sensorRange;
  let frontLethalType = 'none';
  let frontLethalScale = scale;
  let frontLethalSpeed = 5.9;
  let frontLethalAngle = 0;

  let forwardObstacleDanger = false;
  let leftClearance = 0;
  let rightClearance = 0;
  let leftSelfProximity = 0;
  let rightSelfProximity = 0;
  let encirclingThreatRays = 0;

  for (let r = 0; r < rayCount; r++) {
    const relAngle = ((r / rayCount) * Math.PI * 2) - Math.PI;
    const rayAngle = currentAng + relAngle;
    const cosR = Math.cos(rayAngle);
    const sinR = Math.sin(rayAngle);

    let closestDist = sensorRange;
    let hitType = 'none';
    let hitScale = scale;
    let hitSpeed = 5.9;

    // Outer arena boundary
    const b = relX * cosR + relY * sinR;
    const c = relX * relX + relY * relY - arenaRadius * arenaRadius;
    const disc = b * b - c;
    if (disc >= 0) {
      const tWall = -b + Math.sqrt(disc);
      if (tWall > 0 && tWall < closestDist) {
        closestDist = tWall;
        hitType = 'wall';
        hitScale = scale;
      }
    }

    // Enemy snakes
    if (Array.isArray(snakes)) {
      for (let sIdx = 0; sIdx < snakes.length; sIdx++) {
        const other = snakes[sIdx];
        if (!other || other.dead || other === snake || other.id === snake.id) continue;

        const otherScale = other.sc || 1.0;
        const otherSpeed = (typeof other.sp === 'number' && other.sp > 0) ? other.sp : 5.9;
        const otherRadius = 14 * otherScale;
        const hitThreshold = headRadius + otherRadius + 4;
        const hitThresholdSq = hitThreshold * hitThreshold;

        const dxHead = other.xx - headX;
        const dyHead = other.yy - headY;
        const projHead = dxHead * cosR + dyHead * sinR;
        if (projHead > 0 && projHead < closestDist + hitThreshold) {
          const perpHeadSq = (dxHead * dxHead + dyHead * dyHead) - projHead * projHead;
          if (perpHeadSq < hitThresholdSq) {
            const hitDist = Math.max(0, projHead - Math.sqrt(Math.max(0, hitThresholdSq - perpHeadSq)));
            if (hitDist < closestDist) {
              closestDist = hitDist;
              hitType = 'head';
              hitScale = otherScale;
              hitSpeed = otherSpeed;
            }
          }
        }

        const trackId = other.id !== undefined ? other.id : other;
        const track = enemyTracker ? enemyTracker.get(trackId) : null;
        if (track && (track.vx !== 0 || track.vy !== 0)) {
          const leadHorizon = otherSpeed > 7.0 ? 0.35 : 0.25;
          const predX = other.xx + track.vx * leadHorizon;
          const predY = other.yy + track.vy * leadHorizon;
          const predHitThreshold = headRadius + (otherRadius * 1.25) + 4;
          const predHitThresholdSq = predHitThreshold * predHitThreshold;

          const dxPred = predX - headX;
          const dyPred = predY - headY;
          const projPred = dxPred * cosR + dyPred * sinR;
          if (projPred > 0 && projPred < closestDist + predHitThreshold) {
            const perpPredSq = (dxPred * dxPred + dyPred * dyPred) - projPred * projPred;
            if (perpPredSq < predHitThresholdSq) {
              const hitDist = Math.max(0, projPred - Math.sqrt(Math.max(0, predHitThresholdSq - perpPredSq)));
              if (hitDist < closestDist) {
                closestDist = hitDist;
                hitType = 'head';
                hitScale = otherScale;
                hitSpeed = otherSpeed;
              }
            }
          }
        }

        if (Array.isArray(other.pts)) {
          const segStep = other.pts.length > 250 ? 3 : (other.pts.length > 80 ? 2 : 1);
          for (let pIdx = 0; pIdx < other.pts.length; pIdx += segStep) {
            const pt = other.pts[pIdx];
            if (!pt || pt.dying) continue;
            const dxSeg = pt.xx - headX;
            if (dxSeg > closestDist + hitThreshold || dxSeg < -(closestDist + hitThreshold)) continue;
            const dySeg = pt.yy - headY;
            if (dySeg > closestDist + hitThreshold || dySeg < -(closestDist + hitThreshold)) continue;
            const projSeg = dxSeg * cosR + dySeg * sinR;
            if (projSeg > 0 && projSeg < closestDist + hitThreshold) {
              const perpSegSq = (dxSeg * dxSeg + dySeg * dySeg) - projSeg * projSeg;
              if (perpSegSq < hitThresholdSq) {
                const hitDist = Math.max(0, projSeg - Math.sqrt(Math.max(0, hitThresholdSq - perpSegSq)));
                if (hitDist < closestDist) {
                  closestDist = hitDist;
                  hitType = 'body';
                  hitScale = otherScale;
                  hitSpeed = otherSpeed;
                }
              }
            }
          }
        }
      }
    }

    if (Array.isArray(snake.pts)) {
      const selfRadius = headRadius;
      const selfHitThreshold = headRadius + selfRadius + 2;
      const selfHitThresholdSq = selfHitThreshold * selfHitThreshold;
      for (let pIdx = 18; pIdx < snake.pts.length; pIdx += 2) {
        const pt = snake.pts[pIdx];
        if (!pt || pt.dying) continue;
        const dxSelf = pt.xx - headX;
        if (dxSelf > closestDist + selfHitThreshold || dxSelf < -(closestDist + selfHitThreshold)) continue;
        const dySelf = pt.yy - headY;
        if (dySelf > closestDist + selfHitThreshold || dySelf < -(closestDist + selfHitThreshold)) continue;
        const projSelf = dxSelf * cosR + dySelf * sinR;
        if (projSelf > 0 && projSelf < closestDist + selfHitThreshold) {
          const perpSelfSq = (dxSelf * dxSelf + dySelf * dySelf) - projSelf * projSelf;
          if (perpSelfSq < selfHitThresholdSq) {
            const hitDist = Math.max(0, projSelf - Math.sqrt(Math.max(0, selfHitThresholdSq - perpSelfSq)));
            if (hitDist < closestDist) {
              closestDist = hitDist;
              hitType = 'self';
              hitScale = scale;
            }
          }
        }
      }
    }

    const normDist = Math.max(0, Math.min(1.0, closestDist / sensorRange));
    inputs[r] = 1.0 - normDist;

    if (closestDist < minObstacleDist) {
      minObstacleDist = closestDist;
      closestObstacleType = hitType;
      closestObstacleAngle = relAngle;
    }

    if (Math.abs(relAngle) <= Math.PI / 4 && closestDist < frontObstacleDist) {
      frontObstacleDist = closestDist;
      frontThreatScale = hitScale;
      frontObstacleType = hitType;
      frontThreatSpeed = hitSpeed;
      frontObstacleAngle = relAngle;
    }

    if (hitType !== 'self' && hitType !== 'none') {
      if (closestDist < minLethalDist) {
        minLethalDist = closestDist;
        closestLethalType = hitType;
        closestLethalScale = hitScale;
        closestLethalSpeed = hitSpeed;
        closestLethalAngle = relAngle;
      }
      if (Math.abs(relAngle) <= Math.PI / 4 && closestDist < frontLethalDist) {
        frontLethalDist = closestDist;
        frontLethalScale = hitScale;
        frontLethalType = hitType;
        frontLethalSpeed = hitSpeed;
        frontLethalAngle = relAngle;
      }
    }

    if ((hitType === 'body' || hitType === 'head') && closestDist < sensorRange * 0.85) {
      encirclingThreatRays++;
    }

    const degFromHead = Math.abs(relAngle * 180 / Math.PI);
    const obstacleThreshold = (headRadius + (hitScale * 14)) * 2.1;
    if (degFromHead <= 35 && closestDist < obstacleThreshold && hitType !== 'self') {
      forwardObstacleDanger = true;
    }

    if (relAngle > 0 && relAngle < Math.PI * 0.85) {
      leftClearance += closestDist;
    } else if (relAngle < 0 && relAngle > -Math.PI * 0.85) {
      rightClearance += closestDist;
    }

    if (hitType === 'self') {
      const proximity = 1.0 - normDist;
      if (relAngle > 0.15 && relAngle < Math.PI * 0.85) {
        leftSelfProximity += proximity;
      } else if (relAngle < -0.15 && relAngle > -Math.PI * 0.85) {
        rightSelfProximity += proximity;
      }
    }

    activeRays.push({
      angle: rayAngle,
      dist: closestDist,
      norm: normDist,
      hitType
    });
  }

  let selfShieldSide = 0;
  if (leftSelfProximity > 0.25 && leftSelfProximity > rightSelfProximity * 1.35) {
    selfShieldSide = 1;
  } else if (rightSelfProximity > 0.25 && rightSelfProximity > leftSelfProximity * 1.35) {
    selfShieldSide = -1;
  }

  return {
    inputs,
    activeRays,
    minObstacleDist,
    frontObstacleDist,
    frontThreatScale,
    frontThreatSpeed,
    closestObstacleType,
    frontObstacleType,
    closestObstacleAngle,
    frontObstacleAngle,
    minLethalDist,
    closestLethalType,
    closestLethalScale,
    closestLethalSpeed,
    closestLethalAngle,
    frontLethalDist,
    frontLethalType,
    frontLethalScale,
    frontLethalSpeed,
    frontLethalAngle,
    forwardObstacleDanger,
    leftClearance,
    rightClearance,
    selfShieldSide,
    encirclingThreatRays
  };
}

export function findEscapeCorridors(activeRays, headRadius, currentAng, encirclingThreatRays, rayCount = 16) {
  let encirclementEscape = null;
  if (encirclingThreatRays >= 5) {
    let maxSlitClearance = 0;
    let bestSlitAngle = currentAng;
    for (let r = 0; r < rayCount; r++) {
      const ray = activeRays[r];
      if (ray && ray.hitType !== 'wall' && ray.dist > maxSlitClearance) {
        maxSlitClearance = ray.dist;
        bestSlitAngle = ray.angle;
      }
    }
    encirclementEscape = {
      threatRays: encirclingThreatRays,
      slitDist: maxSlitClearance,
      slitAngle: bestSlitAngle,
      isOpen: maxSlitClearance > headRadius * 2.5,
      isEarly: encirclingThreatRays < 7
    };
  }

  let maxCorridorDist = 0;
  let bestCorridorAngle = currentAng;
  for (let r = 0; r < rayCount; r++) {
    const ray = activeRays[r];
    if (ray && ray.hitType !== 'wall' && ray.dist > maxCorridorDist) {
      maxCorridorDist = ray.dist;
      bestCorridorAngle = ray.angle;
    }
  }

  return { encirclementEscape, maxCorridorDist, bestCorridorAngle };
}

export function extractFoodRadar({
  headX,
  headY,
  currentAng,
  foods,
  preys,
  sensorRange,
  foodSectorCount = 8,
  mySpeed = 5.9
}) {
  const foodSectors = new Float32Array(foodSectorCount);
  const detectedFoods = [];
  const foodSectorSpan = (Math.PI * 2) / foodSectorCount;
  const maxNormalFoodDist = Math.max(700, sensorRange * 1.35);
  const maxDebrisDist = 1200;
  let corpsePelletCount = 0;

  let replacePointer = 0;
  const processPellet = (fx, fy, size, isPrey = false) => {
    const isHighValue = isPrey || size >= 6.5;
    const maxDist = isHighValue ? maxDebrisDist : maxNormalFoodDist;
    const dx = fx - headX;
    if (dx > maxDist || dx < -maxDist) return;
    const dy = fy - headY;
    if (dy > maxDist || dy < -maxDist) return;
    const distSq = dx * dx + dy * dy;
    if (distSq > maxDist * maxDist || distSq < 4) return;
    const dist = Math.sqrt(distSq);

    if (!isPrey && size >= 6.5 && dist < 750) {
      corpsePelletCount++;
    }

    let pelletAngle = Math.atan2(dy, dx) - currentAng;
    while (pelletAngle < 0) pelletAngle += Math.PI * 2;
    while (pelletAngle >= Math.PI * 2) pelletAngle -= Math.PI * 2;

    const sectorIdx = Math.floor(pelletAngle / foodSectorSpan) % foodSectorCount;
    const multiplier = isPrey ? 15.0 : (size >= 6.5 ? 10.0 : 1.0);
    const weight = (size * multiplier) / (dist * 0.04 + 1);
    foodSectors[sectorIdx] += weight;

    if (detectedFoods.length < 250) {
      detectedFoods.push({ dx, dy, dist, sz: size, isPrey });
    } else if (isHighValue && replacePointer < 250) {
      while (replacePointer < 250 && (detectedFoods[replacePointer].isPrey || detectedFoods[replacePointer].sz >= 6.5)) {
        replacePointer++;
      }
      if (replacePointer < 250) {
        detectedFoods[replacePointer++] = { dx, dy, dist, sz: size, isPrey };
      }
    }
  };

  const foodList = Array.isArray(foods) ? foods : [];
  for (let i = 0; i < foodList.length; i++) {
    const f = foodList[i];
    if (!f || f.eaten || f.dying) continue;
    const fx = typeof f.xx === 'number' ? f.xx : (typeof f.x === 'number' ? f.x : NaN);
    const fy = typeof f.yy === 'number' ? f.yy : (typeof f.y === 'number' ? f.y : NaN);
    if (!isNaN(fx) && !isNaN(fy)) {
      const sz = f.sz || f.rad || 4;
      processPellet(fx, fy, sz, false);
    }
  }

  const preyList = Array.isArray(preys) ? preys : [];
  for (let i = 0; i < preyList.length; i++) {
    const p = preyList[i];
    if (!p || p.eaten || p.dying) continue;
    const px = typeof p.xx === 'number' ? p.xx : (typeof p.x === 'number' ? p.x : NaN);
    const py = typeof p.yy === 'number' ? p.yy : (typeof p.y === 'number' ? p.y : NaN);
    if (!isNaN(px) && !isNaN(py)) {
      const sz = p.sz || p.rad || 25;
      let targetPx = px;
      let targetPy = py;
      const pSpeed = (typeof p.sp === 'number' && p.sp > 0) ? p.sp : 0;
      const pAng = (typeof p.ang === 'number') ? p.ang : ((typeof p.dir === 'number') ? p.dir : null);

      if (pSpeed > 0 && pAng !== null) {
        const dxPrey = px - headX;
        const dyPrey = py - headY;
        const distPrey = Math.hypot(dxPrey, dyPrey);
        const sprintSpeedSec = mySpeed * 1.85 * 60;
        const leadSec = Math.max(0.10, Math.min(0.60, distPrey / (sprintSpeedSec + 1e-3)));
        const preySpeedSec = pSpeed * 60;
        targetPx = px + Math.cos(pAng) * (preySpeedSec * leadSec);
        targetPy = py + Math.sin(pAng) * (preySpeedSec * leadSec);
      }
      processPellet(targetPx, targetPy, sz, true);
    }
  }

  if (detectedFoods.length > 80) {
    detectedFoods.sort((a, b) => {
      const valA = a.isPrey ? 25 : (a.sz >= 6.5 ? 12 : 1);
      const valB = b.isPrey ? 25 : (b.sz >= 6.5 ? 12 : 1);
      return (a.dist / Math.sqrt(valA)) - (b.dist / Math.sqrt(valB));
    });
    detectedFoods.length = 80;
  }

  const foodInputs = new Float32Array(foodSectorCount);
  for (let f = 0; f < foodSectorCount; f++) {
    foodInputs[f] = Math.min(1.0, foodSectors[f] / 8.0);
  }

  return { foodSectors, detectedFoods, foodInputs, corpsePelletCount };
}

export function pickFoodTarget({
  detectedFoods,
  currentAng,
  headX = 0,
  headY = 0,
  snakes = null,
  lastKillEpicenter = null,
  vacuumActiveUntil = 0,
  slipstreamTarget = null,
  enemyTracker = null,
  now = performance.now()
}) {
  const list = detectedFoods;
  if (!list || list.length === 0) return null;

  const isVacuumActive = Boolean(lastKillEpicenter && now < vacuumActiveUntil);

  const corpsePellets = [];
  for (let i = 0; i < list.length; i++) {
    if (list[i].sz >= 6.5 && !list[i].isPrey) corpsePellets.push(list[i]);
  }

  let best = null;
  let bestCost = Infinity;
  for (let i = 0; i < list.length; i++) {
    const f = list[i];
    const isCorpse = !f.isPrey && f.sz >= 6.5;
    const isHighValue = f.isPrey || isCorpse;
    const maxTargetDist = (isHighValue || isVacuumActive) ? 1200 : 650;
    if (f.dist < 4 || f.dist > maxTargetDist) continue;

    let error = Math.atan2(f.dy, f.dx) - currentAng;
    while (error > Math.PI) error -= Math.PI * 2;
    while (error < -Math.PI) error += Math.PI * 2;

    let value = f.isPrey ? 25 : (isCorpse ? 8 : 1);

    let clusterCount = 0;
    if (isCorpse && corpsePellets.length > 1) {
      for (let j = 0; j < corpsePellets.length; j++) {
        const otherF = corpsePellets[j];
        if (otherF === f) continue;
        const cdx = otherF.dx - f.dx;
        if (cdx > 160 || cdx < -160) continue;
        const cdy = otherF.dy - f.dy;
        if (cdy > 160 || cdy < -160) continue;
        if (cdx * cdx + cdy * cdy < 25600) {
          clusterCount++;
        }
      }
      value *= (1 + Math.min(4.5, clusterCount * 0.50));
    }

    let nearKillEpicenter = false;
    if (isVacuumActive && lastKillEpicenter) {
      const worldFx = headX + f.dx;
      const worldFy = headY + f.dy;
      const distToEpicenter = Math.hypot(worldFx - lastKillEpicenter.x, worldFy - lastKillEpicenter.y);
      if (distToEpicenter < 480 || isCorpse || f.dist < 480) {
        nearKillEpicenter = true;
        value *= 3.0;
      }
    }

    if (slipstreamTarget && f.dist < 450 && Math.abs(error) < 0.6) {
      value *= 2.0;
    }

    let contestedByRival = false;
    let rivalBearingSign = 0;
    if (isHighValue) {
      const worldFx = headX + f.dx;
      const worldFy = headY + f.dy;
      let checkedEnemy = false;
      if (enemyTracker && enemyTracker.size > 0) {
        for (const [, track] of enemyTracker.entries()) {
          const enemyDistToFood = Math.hypot(track.xx - worldFx, track.yy - worldFy);
          if (enemyDistToFood < 240) {
            contestedByRival = true;
            let rBear = Math.atan2(track.yy - headY, track.xx - headX) - currentAng;
            while (rBear > Math.PI) rBear -= Math.PI * 2;
            while (rBear < -Math.PI) rBear += Math.PI * 2;
            rivalBearingSign = rBear >= 0 ? 1 : -1;
            if (enemyDistToFood < f.dist * 0.90 && enemyDistToFood < 170) {
              value *= 0.20;
            } else {
              value *= 0.60;
            }
            checkedEnemy = true;
            break;
          }
        }
      }
      if (!checkedEnemy && Array.isArray(snakes)) {
        for (let sIdx = 0; sIdx < snakes.length; sIdx++) {
          const other = snakes[sIdx];
          if (!other || other.dead || (other.id !== undefined && other.id === 'me')) continue;
          const ox = typeof other.xx === 'number' ? other.xx : other.x;
          const oy = typeof other.yy === 'number' ? other.yy : other.y;
          if (typeof ox !== 'number' || typeof oy !== 'number') continue;
          const enemyDistToFood = Math.hypot(ox - worldFx, oy - worldFy);
          if (enemyDistToFood < 240) {
            contestedByRival = true;
            let rBear = Math.atan2(oy - headY, ox - headX) - currentAng;
            while (rBear > Math.PI) rBear -= Math.PI * 2;
            while (rBear < -Math.PI) rBear += Math.PI * 2;
            rivalBearingSign = rBear >= 0 ? 1 : -1;
            if (enemyDistToFood < f.dist * 0.90 && enemyDistToFood < 170) {
              value *= 0.20;
            } else {
              value *= 0.60;
            }
            break;
          }
        }
      }
    }

    const turnWeight = (nearKillEpicenter || clusterCount >= 3) ? 0.55 : 2.2;
    const turnPenalty = 1 + turnWeight * (1 - Math.cos(error));
    const cost = ((f.dist + 50) * turnPenalty) / value;
    if (cost < bestCost) {
      bestCost = cost;
      best = {
        error,
        dist: f.dist,
        value,
        dx: f.dx,
        dy: f.dy,
        sz: f.sz,
        isPrey: f.isPrey,
        isCorpse,
        clusterCount,
        contestedByRival,
        rivalBearingSign
      };
    }
  }
  return best;
}
