import React, { useRef, useEffect, useState, useCallback } from "react";
import Human from "@vladmandic/human";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import backgroundVideo from "./assets/videos/background.mp4";
import "./index.scss";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const humanRef = useRef(null);
  const detectorRef = useRef(null);
  const streamRef = useRef(null);
  const offscreenMask = useRef(document.createElement("canvas"));

  const [started, setStarted] = useState(false);
  const [canvasStyle, setCanvasStyle] = useState({});

  const lastBoxes = useRef([]);
  const targetBoxes = useRef([]);
  const lastUpdateTime = useRef(0);
  const rawDetections = useRef([]);

  const personColors = useRef(new Map());
  const nextPersonIndex = useRef(0);
  const nextId = useRef(Math.floor(10000 + Math.random() * 10000));

  const drawMultilineText = (ctx, text, x, y, lineHeight = 18) => {
    const lines = text.split('\n');
    lines.forEach((line, i) => {
      ctx.fillText(line, x, y + i * lineHeight);
    });
  };

  // === Init Human + Detector ===
  const initModels = useCallback(async () => {
    const human = new Human({
      backend: "webgl",
      async: true,
      segmentation: {
        enabled: true,
        modelPath: "./models/rvm.json",
        return: "mask",
        smooth: false,
      },
      body: { enabled: true },
      face: { enabled: false },
      hand: { enabled: false },
      object: { enabled: false },
    });
    await human.load();
    humanRef.current = human;
    console.log("✅ Human (RVM) loaded");

    const detector = await cocoSsd.load();
    detectorRef.current = detector;
    console.log("✅ COCO-SSD loaded");
  }, []);

  // === Start camera ===
  const startCamera = useCallback(async (onReady) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        if (canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          ctxRef.current = canvasRef.current.getContext("2d");
          offscreenMask.current.width = videoRef.current.videoWidth;
          offscreenMask.current.height = videoRef.current.videoHeight;
        }
        onReady();
      };
    } catch (err) {
      console.error("Camera error:", err);
    }
  }, []);

  // === Segmentation loop ===
  const segmentLoop = useCallback(async () => {
    const vid = videoRef.current;
    const human = humanRef.current;
    const offCtx = offscreenMask.current?.getContext("2d");
    if (!vid || !human || !offCtx) return;

    const segTensor = await human.segmentation(vid);
    if (segTensor) {
      const [h, w] = segTensor.shape;
      const rgba = await segTensor.data();
      const curr = new Uint8ClampedArray(rgba);

      // base cyan tint
      for (let i = 0; i < curr.length; i += 4) {
        const alpha = curr[i + 3];
        curr[i] = 0;
        curr[i + 1] = 200;
        curr[i + 2] = 200;
        curr[i + 3] = alpha > 50 ? 200 : 0;
      }

      offCtx.clearRect(0, 0, w, h);
      offCtx.putImageData(new ImageData(curr, w, h), 0, 0);

      // localized tint for red people
      const redPeople = lastBoxes.current.filter(
        (b) => personColors.current.get(b.id) === "red"
      );
      if (redPeople.length > 0) {
        offCtx.save();
        offCtx.globalCompositeOperation = "source-atop";
        const now = performance.now() / 1000;
        const flashPhase = (Math.sin(now * Math.PI * 2) + 1) / 2;
        const intensity = 100 + Math.floor(flashPhase * 155);
        offCtx.fillStyle = `rgba(${intensity},0,0,1)`;
        redPeople.forEach((b) => offCtx.fillRect(b.x, b.y, b.w, b.h));
        offCtx.restore();

        // take the first red person as the zoom target
        const target = redPeople[0];
        const cx = target.x + target.w / 2;
        const cy = target.y;
        setCanvasStyle({
          transform: "scale(1.5)",
          transformOrigin: `${cx}px ${cy}px`,
        });
      } else {
        setCanvasStyle({ transform: "scale(1))" });
      }

      // === outlines + labels ===
      offCtx.save();
      offCtx.lineWidth = 4;
      offCtx.font = "bold 16px monospace";
      offCtx.textBaseline = "bottom";

      lastBoxes.current.forEach((b) => {
        const color = personColors.current.get(b.id) || "cyan";

        if (color === "red") {
          offCtx.strokeStyle = "rgb(255,0,0)";
          offCtx.fillStyle = "rgb(255,0,0)";
          offCtx.strokeRect(b.x, b.y, b.w, b.h);
          drawMultilineText(offCtx,
            `person ${String(b.id).padStart(6, "0")}:\nENEMY OF THE STATE!!!`,
            b.x - 3,
            b.y - 18
          );
        } else {
          offCtx.strokeStyle = "rgba(4,236,255,1)";
          offCtx.fillStyle = "rgba(4,236,255,1)";
          offCtx.strokeRect(b.x, b.y, b.w, b.h);
          drawMultilineText(offCtx,
            `person ${String(b.id).padStart(6, "0")}:\nThreat level: low`,
            b.x - 3,
            b.y - 18
          );
        }
      });

      offCtx.restore();
      segTensor.dispose();
    }

    setTimeout(segmentLoop, 100);
  }, []);

  // === Person detection loop ===
  const detectLoop = useCallback(async () => {
    const vid = videoRef.current;
    const detector = detectorRef.current;
    if (!vid || !detector) return;

    try {
      const results = await detector.detect(vid);
      rawDetections.current = results;

      const detections = results
        .filter((r) => r.class === "person" && r.score > 0.5)
        .map((r) => ({
          x: r.bbox[0],
          y: r.bbox[1],
          w: r.bbox[2],
          h: r.bbox[3],
          id: null,
          notFoundFrames: 0,
        }));

      const prev = targetBoxes.current;
      const newBoxes = [];

      const centerDist2 = (a, b) => {
        const ax = a.x + a.w / 2,
          ay = a.y + a.h / 2;
        const bx = b.x + b.w / 2,
          by = b.y + b.h / 2;
        const dx = ax - bx,
          dy = ay - by;
        return dx * dx + dy * dy;
      };

      const usedPrev = new Set();
      for (const det of detections) {
        let bestIdx = -1;
        let bestD2 = Infinity;

        prev.forEach((p, i) => {
          if (usedPrev.has(i)) return;
          const d2 = centerDist2(p, det);
          if (d2 < bestD2) {
            bestD2 = d2;
            bestIdx = i;
          }
        });

        if (bestIdx !== -1 && Math.sqrt(bestD2) < 400) {
          const prevBox = prev[bestIdx];
          det.id = prevBox.id ?? nextId.current++;
          det.notFoundFrames = 0;
          usedPrev.add(bestIdx);
        } else {
          det.id = nextId.current++;
          det.notFoundFrames = 0;
        }

        if (!personColors.current.has(det.id)) {
          nextPersonIndex.current += 1;
          const isRed = nextPersonIndex.current % 5 === 0;
          personColors.current.set(det.id, isRed ? "red" : "cyan");
        }

        newBoxes.push(det);
      }

      // increment missed count for unmatched previous boxes
      const survivors = [];
      prev.forEach((p) => {
        const match = newBoxes.find((b) => b.id === p.id);
        if (!match) {
          const missed = { ...p, notFoundFrames: (p.notFoundFrames ?? 0) + 1 };
          if (missed.notFoundFrames <= 5) survivors.push(missed);
          else personColors.current.delete(p.id);
        }
      });

      targetBoxes.current = [...newBoxes, ...survivors];

      lastBoxes.current = lastBoxes.current.filter((b) =>
        targetBoxes.current.find((t) => t.id === b.id)
      );

      if (lastBoxes.current.length === 0) {
        lastBoxes.current = targetBoxes.current.map((b) => ({ ...b }));
      }

      lastUpdateTime.current = performance.now();
    } catch (err) {
      console.warn("Detection failed:", err);
    }

    setTimeout(detectLoop, 250);
  }, []);

  // === Draw loop ===
  const drawLoop = useCallback(() => {
    const vid = videoRef.current;
    const ctx = ctxRef.current;
    if (!vid || !ctx) return;

    ctx.globalAlpha = 1;
    ctx.filter = "none";
    ctx.drawImage(vid, 0, 0, ctx.canvas.width, ctx.canvas.height);

    if (offscreenMask.current) {
      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.drawImage(offscreenMask.current, 0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.restore();
    }

    const lerpFactor = 0.1;

    // 1) lerp existing ones
    lastBoxes.current = lastBoxes.current.map((b) => {
      const t = targetBoxes.current.find((x) => x.id === b.id);
      if (!t) return b;
      return {
        ...b,
        x: b.x + (t.x - b.x) * lerpFactor,
        y: b.y + (t.y - b.y) * lerpFactor,
        w: b.w + (t.w - b.w) * lerpFactor,
        h: b.h + (t.h - b.h) * lerpFactor,
        notFoundFrames: t.notFoundFrames,
      };
    });

    // 2) add new tracked boxes if missing
    targetBoxes.current.forEach((t) => {
      const exists = lastBoxes.current.find((b) => b.id === t.id);
      if (!exists) lastBoxes.current.push({ ...t });
    });

    vid.requestVideoFrameCallback(drawLoop);
  }, []);

  // === Lifecycle ===
  useEffect(() => {
    let active = true;
    const start = async () => {
      await initModels();
      if (!active) return;
      await startCamera(() => {
        setStarted(true);
        videoRef.current.requestVideoFrameCallback(drawLoop);
        segmentLoop();
        detectLoop();
      });
    };
    start();
    return () => {
      active = false;
      if (streamRef.current)
        streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, [initModels, startCamera, drawLoop, segmentLoop, detectLoop]);

  return (
    <div className="app">
      <video className="background" playsInline muted autoPlay loop>
        <source src={backgroundVideo} />
      </video>
      <video className="video" ref={videoRef} playsInline muted />
      <div className="camera">
        <canvas ref={canvasRef} className="camera-canvas" style={canvasStyle} />
      </div>
      {!started && <div className="loading">Loading models and camera...</div>}
    </div>
  );
};

export default App;
