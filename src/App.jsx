import React, { useRef, useEffect, useState, useCallback } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import { Pose } from "@mediapipe/pose";
import { matchTracks, resetTrackIds } from "./utils";
import "./index.scss";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const detectorRef = useRef(null);
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const prevTracks = useRef([]);

  const [started, setStarted] = useState(false);

  // === Helper: stable color per ID ===
  const colorForId = (id) => {
    let hash = 0;
    for (let i = 0; i < id.length; i++) hash = id.charCodeAt(i) + ((hash << 5) - hash);
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 100%, 65%)`;
  };

  // === Init models ===
  const initModels = useCallback(async () => {
    const nextDetector = await cocoSsd.load();
    detectorRef.current = nextDetector;

    const poseDetector = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
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
        poseDetector.onResults((r) => resolve(r));
        poseDetector.send({ image });
      });

    poseRef.current = { instance: poseDetector, runPose };
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

  const frameCounter = useRef(0);
  const lastMasks = useRef({});

  const processFrame = useCallback(async (now) => {
    const v = videoRef.current;
    const ctx = ctxRef.current;
    const detector = detectorRef.current;
    const pose = poseRef.current;

    if (!v || !ctx || !detector || !pose) return;

    const detections = await detector.detect(v);
    const people = detections
      .filter((d) => d.class === "person" && d.score > 0.5)
      .map((d) => ({
        x: d.bbox[0],
        y: d.bbox[1],
        w: d.bbox[2],
        h: d.bbox[3],
      }))
      .slice(0, 5); // limit to top 5

    const tracked = matchTracks(people, prevTracks.current);
    prevTracks.current = tracked;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(v, 0, 0, ctx.canvas.width, ctx.canvas.height);

    frameCounter.current++;

    for (const t of tracked) {
      const color = "rgba(0,255,255,0.5)"; // cyan overlay
      const shouldRun = frameCounter.current % 3 === 0;
      let results = lastMasks.current[t.id];

      // expand box before running segmentation
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

      // Draw segmentation mask if available
      if (results?.segmentationMask) {
        // Create tinted mask
        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = w;
        maskCanvas.height = h;
        const mctx = maskCanvas.getContext("2d");

        // draw segmentation mask to offscreen canvas
        mctx.drawImage(results.segmentationMask, 0, 0, w, h);

        // apply cyan tint
        const maskData = mctx.getImageData(0, 0, w, h);
        const d = maskData.data;
        for (let i = 0; i < d.length; i += 4) {
          const alpha = d[i]; // segmentation mask intensity
          d[i] = 0;           // R
          d[i + 1] = 255;     // G
          d[i + 2] = 255;     // B
          d[i + 3] = alpha * 0.5; // 50% opacity
        }
        mctx.putImageData(maskData, 0, 0);

        // draw tinted mask back to main canvas
        ctx.drawImage(maskCanvas, x, y, w, h);

        // draw cyan outline around the mask itself
        // ctx.save();
        // ctx.strokeStyle = "cyan";
        // ctx.lineWidth = 3;
        // ctx.shadowColor = "rgba(0,255,255,0.6)";
        // ctx.shadowBlur = 5;
        // ctx.strokeRect(x, y, w, h);
        // ctx.restore();
      }

      // optional bounding box (for debugging)
      ctx.strokeStyle = "rgba(0,255,255,0.2)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, w, h);
    }

    v.requestVideoFrameCallback(processFrame);
  }, []);


  // === Lifecycle ===
  useEffect(() => {
    let active = true;
    resetTrackIds();

    const start = async () => {
      await initModels();
      if (!active) return;
      console.log("✅ Models initialized");
      await startCamera(() => {
        setStarted(true);
        videoRef.current.requestVideoFrameCallback(processFrame);
      });
    };

    start();

    return () => {
      active = false;

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      const vid = videoRef.current;
      if (vid) {
        try {
          vid.srcObject = null;
          vid.pause();
        } catch { }
      }
    };
  }, [initModels, startCamera, processFrame]);

  return (
    <div className="app">
      <video className="video" ref={videoRef} playsInline muted />
      <canvas className="canvas" ref={canvasRef} />
      {!started && <div className="loading">Loading models and camera...</div>}
    </div>
  );
};

export default App;
