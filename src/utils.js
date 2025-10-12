// utils/tracking.js

// ---- Tunables ----
const IOU_THRESHOLD = 0.2;   // how much overlap to accept a match
const MAX_MISSES = 10;    // frames to keep a track alive w/o matches
const SMOOTH_POS = 0.6;   // 0..1; higher = follow detections more
const SMOOTH_SIZE = 0.6;   // smoothing for w/h
const DILATE_PRED = 1.05;  // inflate predicted box a bit to help matching

// ---- ID generator (sequential) ----
let nextId = 1;
const genId = () => String(nextId++).padStart(4, '0');



export const resetTrackIds = () => { nextId = 1; };

// ---- IoU on {x,y,w,h} ----
const iou = (a, b) => {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
};

// ---- Predict next box with constant velocity ----
const predictBox = (t) => {
  const x = t.x + (t.vx ?? 0);
  const y = t.y + (t.vy ?? 0);
  const w = t.w;
  const h = t.h;
  // Slight dilation to tolerate detector jitter
  const cx = x + w / 2, cy = y + h / 2;
  const w2 = w * DILATE_PRED, h2 = h * DILATE_PRED;
  return { x: cx - w2 / 2, y: cy - h2 / 2, w: w2, h: h2 };
};

// ---- Greedy assign detections to predicted tracks by IoU ----
const assignDetections = (preds, dets) => {
  const assignments = new Array(dets.length).fill(-1); // det -> track index
  const usedTracks = new Set();

  for (let di = 0; di < dets.length; di++) {
    let bestTi = -1, bestIoU = 0;
    for (let ti = 0; ti < preds.length; ti++) {
      if (usedTracks.has(ti)) continue;
      const j = iou(preds[ti], dets[di]);
      if (j > bestIoU) { bestIoU = j; bestTi = ti; }
    }
    if (bestIoU >= IOU_THRESHOLD && bestTi !== -1) {
      assignments[di] = bestTi;
      usedTracks.add(bestTi);
    }
  }
  return assignments; // -1 = new track
};

/**
 * detections: Array<{x,y,w,h}>
 * prevTracks: Array<Track>
 * Track schema we maintain internally:
 * { id, x,y,w,h, vx,vy, age, missed }
 */
export const matchTracks = (detections = [], prevTracks = []) => {
  // 1) Predict step for all existing tracks
  const predictions = prevTracks.map(predictBox);

  // 2) Assign detections to predicted tracks
  const assignments = assignDetections(predictions, detections);

  // 3) Build updated track list
  const updated = [];

  // a) Matched detections → update position, velocity, reset missed
  assignments.forEach((ti, di) => {
    const det = detections[di];
    if (ti === -1) return; // new track; handle later

    const tPrev = prevTracks[ti];

    // velocity from last raw pos to new detection
    const vx = det.x - tPrev.x;
    const vy = det.y - tPrev.y;

    // smooth update (EMA) for stability
    const x = SMOOTH_POS * det.x + (1 - SMOOTH_POS) * (tPrev.x + (tPrev.vx ?? 0));
    const y = SMOOTH_POS * det.y + (1 - SMOOTH_POS) * (tPrev.y + (tPrev.vy ?? 0));
    const w = SMOOTH_SIZE * det.w + (1 - SMOOTH_SIZE) * tPrev.w;
    const h = SMOOTH_SIZE * det.h + (1 - SMOOTH_SIZE) * tPrev.h;

    updated.push({
      id: tPrev.id,
      x, y, w, h,
      vx, vy,
      age: (tPrev.age ?? 0) + 1,
      missed: 0,
    });
  });

  // b) Unmatched detections → spawn new tracks with sequential IDs
  detections.forEach((det, di) => {
    if (assignments[di] !== -1) return; // already matched
    const id = genId();
    console.log('new match', id)
    updated.push({
      id,
      x: det.x, y: det.y, w: det.w, h: det.h,
      vx: 0, vy: 0,
      age: 0,
      missed: 0,
    });
  });

  // c) Unmatched previous tracks → age & keep until MAX_MISSES
  prevTracks.forEach((tPrev, ti) => {
    const stillPresent = updated.findIndex(t => t.id === tPrev.id) !== -1;
    if (stillPresent) return;

    const pred = predictBox(tPrev); // drift forward once while missing
    const missed = (tPrev.missed ?? 0) + 1;
    if (missed <= MAX_MISSES) {
      updated.push({
        id: tPrev.id,
        x: pred.x, y: pred.y, w: tPrev.w, h: tPrev.h,
        vx: tPrev.vx ?? 0, vy: tPrev.vy ?? 0,
        age: (tPrev.age ?? 0) + 1,
        missed,
      });
    }
  });

  return updated;
};
