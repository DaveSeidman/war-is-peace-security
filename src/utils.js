import { v4 as uuidv4 } from 'uuid';

// Compute bounding boxes from segmentation masks
const getBBoxes = segmentations =>
  segmentations.map(s => {
    const mask = s.mask;
    let minX = mask.width, minY = mask.height, maxX = 0, maxY = 0;
    const data = mask.data;
    for (let i = 0; i < data?.length; i++) {
      if (data[i] > 0) {
        const x = i % mask.width;
        const y = Math.floor(i / mask.width);
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
      id: null
    };
  });

// Compute IoU between two boxes
const iou = (a, b) => {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return inter / union;
};

// Match current detections to existing tracks
export const matchTracks = (segmentations, prevTracks = []) => {
  const boxes = getBBoxes(segmentations);
  const newTracks = [];

  boxes.forEach(box => {
    let bestMatch = null;
    let bestIoU = 0;

    prevTracks.forEach(track => {
      const i = iou(track.bbox, box);
      if (i > bestIoU && i > 0.3) { // threshold
        bestIoU = i;
        bestMatch = track;
      }
    });

    if (bestMatch) {
      newTracks.push({ id: bestMatch.id, bbox: box });
    } else {
      newTracks.push({ id: uuidv4().slice(0, 4), bbox: box });
    }
  });

  return newTracks;
};
