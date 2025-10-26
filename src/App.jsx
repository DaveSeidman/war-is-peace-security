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
  const frameCounter = useRef(0);
  const lastMasks = useRef({});

  const [started, setStarted] = useState(false);

  // === Init models ===
  const initModels = useCallback(async () => {
    const nextDetector = await cocoSsd.load();
    detectorRef.current = nextDetector;

    const poseDetector = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    poseDetector.setOptions({
      modelComplexity: 2,
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

  // === Main Frame Loop ===
  const processFrame = useCallback(async (now) => {
    const v = videoRef.current;
    const ctx = ctxRef.current;
    const detector = detectorRef.current;
    const pose = poseRef.current;

    if (!v || !ctx || !detector || !pose) return;

    // --- Step 1: Downsample frame for SSD ---
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
        // Scale coordinates back to full-res
        x: (d.bbox[0] / DOWNSAMPLED_W) * v.videoWidth,
        y: (d.bbox[1] / DOWNSAMPLED_H) * v.videoHeight,
        w: (d.bbox[2] / DOWNSAMPLED_W) * v.videoWidth,
        h: (d.bbox[3] / DOWNSAMPLED_H) * v.videoHeight,
      }))
      .slice(0, 5); // limit to top 5 people

    const tracked = matchTracks(people, prevTracks.current);
    prevTracks.current = tracked;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(v, 0, 0, ctx.canvas.width, ctx.canvas.height);

    frameCounter.current++;

    for (const t of tracked) {
      const shouldRun = frameCounter.current % 3 === 0;
      let results = lastMasks.current[t.id];

      // Expand bounding box
      const pad = 50;
      const x = Math.max(0, t.x - pad);
      const y = Math.max(0, t.y - pad);
      const w = Math.min(v.videoWidth - x, t.w + pad * 2);
      const h = Math.min(v.videoHeight - y, t.h + pad * 2);

      // --- Step 2: Full-res pose detection ---
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

      // === Draw Segmentation Mask ===
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

      // === Draw Pose Landmarks ===
      // if (results?.poseLandmarks) {
      //   ctx.save();
      //   ctx.translate(x, y);
      //   const scaleX = w;
      //   const scaleY = h;
      //   ctx.fillStyle = "rgba(255,255,0,0.9)";
      //   ctx.strokeStyle = "rgba(255,255,0,0.6)";
      //   ctx.lineWidth = 2;

      // const drawConnection = (a, b) => {
      //   const p1 = results.poseLandmarks[a];
      //   const p2 = results.poseLandmarks[b];
      //   if (p1 && p2) {
      //     ctx.beginPath();
      //     ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
      //     ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
      //     ctx.stroke();
      //   }
      // };

      // const connections = [
      //   [11, 13], [13, 15], [12, 14], [14, 16], // arms
      //   [11, 12], [23, 24], [11, 23], [12, 24], // torso
      //   [23, 25], [25, 27], [24, 26], [26, 28], // legs
      // ];
      // connections.forEach(([a, b]) => drawConnection(a, b));

      // for (const lm of results.poseLandmarks) {
      //   ctx.beginPath();
      //   ctx.arc(lm.x * scaleX, lm.y * scaleY, 3, 0, Math.PI * 2);
      //   ctx.fill();
      // }
      //   ctx.restore();
      // }

      // === Bounding Box (optional/debug) ===
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
