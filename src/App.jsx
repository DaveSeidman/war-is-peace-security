import React, { useRef, useEffect, useState, useCallback } from "react";
import Human from "@vladmandic/human";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
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

  // --- smooth animation of bounding boxes ---
  const lastBoxes = useRef([]);
  const targetBoxes = useRef([]);
  const lastUpdateTime = useRef(0);

  // === Init Human + Detector ===
  const initModels = useCallback(async () => {
    // RVM segmentation model
    const human = new Human({
      backend: "webgl",
      async: true,
      modelBasePath: "/models/",
      segmentation: {
        enabled: true,
        modelPath: "/models/rvm.json",
        return: "mask",
        smooth: true,
        threshold: 0.01,
      },
      body: { enabled: false },
      face: { enabled: false },
      hand: { enabled: false },
      object: { enabled: false },
    });
    await human.load();
    humanRef.current = human;
    console.log("✅ Human (RVM) loaded");

    // COCO SSD detector
    const detector = await cocoSsd.load();
    detectorRef.current = detector;
    console.log("✅ COCO-SSD loaded");
  }, []);

  // === Start camera ===
  const startCamera = useCallback(async (onReady) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      const vid = videoRef.current;
      vid.srcObject = stream;
      vid.onloadedmetadata = () => {
        vid.play();
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = vid.videoWidth;
          canvas.height = vid.videoHeight;
          ctxRef.current = canvas.getContext("2d");
          offscreenMask.current.width = vid.videoWidth;
          offscreenMask.current.height = vid.videoHeight;
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
    if (!vid || !human) return;

    const segTensor = await human.segmentation(vid);
    if (segTensor) {
      const [h, w] = segTensor.shape;
      const rgba = await segTensor.data();
      const curr = new Uint8ClampedArray(rgba);

      // cyan tint for humans
      for (let i = 0; i < curr.length; i += 4) {
        const alpha = curr[i + 3];
        curr[i] = 0;
        curr[i + 1] = 200;
        curr[i + 2] = 200;
        curr[i + 3] = alpha > 80 ? alpha : 0;
      }

      const offCtx = offscreenMask.current.getContext("2d");
      offCtx.putImageData(new ImageData(curr, w, h), 0, 0);
      segTensor.dispose();
    }

    setTimeout(segmentLoop, 100); // 10 fps
  }, []);

  // === Person detection loop (non-blocking) ===
  const detectLoop = useCallback(async () => {
    const vid = videoRef.current;
    const detector = detectorRef.current;
    if (!vid || !detector) return;

    try {
      const results = await detector.detect(vid);
      const people = results
        .filter((r) => r.class === "person" && r.score > 0.5)
        .map((r) => ({
          x: r.bbox[0],
          y: r.bbox[1],
          w: r.bbox[2],
          h: r.bbox[3],
        }));

      // Update target boxes
      lastBoxes.current = targetBoxes.current.length
        ? targetBoxes.current.map((b, i) => {
          const n = people[i] || b;
          return {
            x: b.x + (n.x - b.x) * 0.2,
            y: b.y + (n.y - b.y) * 0.2,
            w: b.w + (n.w - b.w) * 0.2,
            h: b.h + (n.h - b.h) * 0.2,
          };
        })
        : people;
      targetBoxes.current = people;
      lastUpdateTime.current = performance.now();
    } catch (err) {
      console.warn("Detection failed:", err);
    }

    setTimeout(detectLoop, 500); // run every ~0.5 s
  }, []);

  // === Draw loop (runs every video frame) ===
  const drawLoop = useCallback(() => {
    const vid = videoRef.current;
    const ctx = ctxRef.current;
    if (!vid || !ctx) return;

    // Draw video
    ctx.globalAlpha = 1;
    ctx.filter = "none";
    ctx.drawImage(vid, 0, 0, ctx.canvas.width, ctx.canvas.height);

    // Draw segmentation mask
    if (offscreenMask.current) {
      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.filter = "blur(2px) contrast(150%) brightness(120%)";
      ctx.drawImage(offscreenMask.current, 0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.restore();
    }

    // Animate bounding boxes
    const lerpFactor = 0.1;
    lastBoxes.current = lastBoxes.current.map((b, i) => {
      const t = targetBoxes.current[i];
      if (!t) return b;
      return {
        x: b.x + (t.x - b.x) * lerpFactor,
        y: b.y + (t.y - b.y) * lerpFactor,
        w: b.w + (t.w - b.w) * lerpFactor,
        h: b.h + (t.h - b.h) * lerpFactor,
      };
    });

    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(255,0,0,0.8)";
    lastBoxes.current.forEach((b) => {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
    });
    ctx.restore();

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
        detectLoop(); // async detection loop
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
      <video
        className="video"
        ref={videoRef}
        playsInline
        muted
      />
      <canvas ref={canvasRef} className="canvas" />
      {!started && <div className="loading">Loading models and camera...</div>}
    </div>
  );
};

export default App;
