import React, { useRef, useEffect, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs"; // ✅ Add TensorFlow import first
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import { Pose } from "@mediapipe/pose";
import { matchTracks, resetTrackIds } from "./utils";
import backgroundVideo from "./assets/videos/background.mp4";
import "./index.scss";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const detectorRef = useRef(null);
  const poseRef = useRef(null);
  const streamRef = useRef(null);

  const prevTracks = useRef([]);
  const frameCounter = useRef(0);
  const lastMasks = useRef({});
  const targetBoxes = useRef([]);
  const lastBoxes = useRef([]);
  const detecting = useRef(false);

  const [started, setStarted] = useState(false);

  // === Init models safely (fix for WebGPU error) ===
  const initModels = useCallback(async () => {
    console.log("🧠 Initializing TensorFlow backend...");
    await tf.setBackend("webgl"); // Force a stable backend
    await tf.ready();
    console.log("✅ TensorFlow backend ready:", tf.getBackend());

    console.log("📦 Loading coco-ssd...");
    const detector = await cocoSsd.load();
    detectorRef.current = detector;
    console.log("✅ coco-ssd model loaded");

    console.log("📦 Loading MediaPipe Pose...");
    const poseDetector = new Pose({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    poseDetector.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: true,
      smoothSegmentation: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    const runPose = (image) =>
      new Promise((resolve) => {
        poseDetector.onResults(resolve);
        poseDetector.send({ image });
      });

    poseRef.current = { instance: poseDetector, runPose };
    console.log("✅ Pose model initialized");
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
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = vid.videoWidth;
          canvas.height = vid.videoHeight;
          ctxRef.current = canvas.getContext("2d");
        }
        vid.play();
        onReady();
      };
    } catch (err) {
      console.error("Camera access error:", err);
    }
  }, []);

  // === Async Detection Loop (~2FPS) ===
  const detectPeople = useCallback(async () => {
    if (detecting.current) return;
    detecting.current = true;

    const v = videoRef.current;
    const detector = detectorRef.current;
    if (!v || !detector) return;

    const DOWNSAMPLED_W = 320;
    const DOWNSAMPLED_H = 180;
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = DOWNSAMPLED_W;
    tempCanvas.height = DOWNSAMPLED_H;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(v, 0, 0, DOWNSAMPLED_W, DOWNSAMPLED_H);

    const detections = await detector.detect(tempCanvas);
    const people = detections
      .filter((d) => d.class === "person" && d.score > 0.5)
      .map((d) => ({
        x: (d.bbox[0] / DOWNSAMPLED_W) * v.videoWidth,
        y: (d.bbox[1] / DOWNSAMPLED_H) * v.videoHeight,
        w: (d.bbox[2] / DOWNSAMPLED_W) * v.videoWidth,
        h: (d.bbox[3] / DOWNSAMPLED_H) * v.videoHeight,
      }))
      .slice(0, 5);

    // Track consistent IDs
    const tracked = matchTracks(people, prevTracks.current);
    prevTracks.current = tracked;

    targetBoxes.current = tracked.map((t, i) => ({
      id: `person-${i + 1}`,
      ...t,
    }));

    if (lastBoxes.current.length === 0)
      lastBoxes.current = targetBoxes.current.map((b) => ({ ...b }));

    detecting.current = false;
    setTimeout(detectPeople, 500); // re-run every half-second
  }, []);

  // === Frame Processing (runs every frame) ===
  const processFrame = useCallback(async () => {
    const v = videoRef.current;
    const ctx = ctxRef.current;
    const pose = poseRef.current;
    if (!v || !ctx || !pose) return;

    // Draw webcam frame
    ctx.globalAlpha = 1;
    ctx.filter = "none";
    ctx.drawImage(v, 0, 0, ctx.canvas.width, ctx.canvas.height);

    // Animate bounding boxes toward targets
    const lerp = (a, b, t) => a + (b - a) * t;
    const newBoxes = lastBoxes.current.map((b, i) => {
      const t = targetBoxes.current[i];
      if (!t) return b;
      return {
        id: t.id,
        x: lerp(b.x, t.x, 0.2),
        y: lerp(b.y, t.y, 0.2),
        w: lerp(b.w, t.w, 0.2),
        h: lerp(b.h, t.h, 0.2),
      };
    });
    lastBoxes.current = targetBoxes.current.length
      ? newBoxes
      : lastBoxes.current;

    // Segmentation every few frames
    frameCounter.current++;
    for (const t of targetBoxes.current) {
      const shouldRun = frameCounter.current % 3 === 0;
      let results = lastMasks.current[t.id];

      const pad = 50;
      const x = Math.max(0, t.x - pad);
      const y = Math.max(0, t.y - pad);
      const w = Math.min(v.videoWidth - x, t.w + pad * 2);
      const h = Math.min(v.videoHeight - y, t.h + pad * 2);

      if (shouldRun) {
        const off = document.createElement("canvas");
        const SEG_W = 256;
        const SEG_H = 256;
        off.width = SEG_W;
        off.height = SEG_H;
        const octx = off.getContext("2d");
        octx.drawImage(v, x, y, w, h, 0, 0, SEG_W, SEG_H);
        results = await pose.runPose(off);
        lastMasks.current[t.id] = results;
      }

      if (results?.segmentationMask) {
        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = w;
        maskCanvas.height = h;
        const mctx = maskCanvas.getContext("2d");
        mctx.drawImage(results.segmentationMask, 0, 0, w, h);

        const maskData = mctx.getImageData(0, 0, w, h);
        const d = maskData.data;
        for (let i = 0; i < d.length; i += 4) {
          const alpha = d[i];
          d[i] = 0;
          d[i + 1] = 255;
          d[i + 2] = 255;
          d[i + 3] = alpha * 0.5;
        }
        mctx.putImageData(maskData, 0, 0);
        ctx.drawImage(maskCanvas, x, y, w, h);
      }
    }

    // === Draw boxes + labels ===
    ctx.save();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(4,236,255,1)";
    ctx.fillStyle = "rgba(4,236,255,0.6)";
    ctx.font = "28px sans-serif";
    ctx.textBaseline = "bottom";

    lastBoxes.current.forEach((b) => {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.fillText(`${b.id}`, b.x + 6, b.y - 6);
    });
    ctx.restore();

    v.requestVideoFrameCallback(processFrame);
  }, []);

  // === Lifecycle ===
  useEffect(() => {
    let active = true;
    resetTrackIds();

    const start = async () => {
      await initModels();
      if (!active) return;
      await startCamera(() => {
        setStarted(true);
        detectPeople();
        videoRef.current.requestVideoFrameCallback(processFrame);
      });
    };

    start();

    return () => {
      active = false;
      if (streamRef.current)
        streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, [initModels, startCamera, detectPeople, processFrame]);

  return (
    <div className="app">
      <video className="background" playsInline muted autoPlay loop>
        <source src={backgroundVideo} />
      </video>

      <video className="video" ref={videoRef} playsInline muted />
      <canvas ref={canvasRef} className="canvas" />
      {!started && <div className="loading">Loading models and camera...</div>}
    </div>
  );
};

export default App;
