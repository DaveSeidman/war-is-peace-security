import React, { useRef, useEffect, useState, useCallback } from "react";
import Human from "@vladmandic/human";
import "./index.scss";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const humanRef = useRef(null);

  const offscreenMask = useRef(document.createElement("canvas"));
  const lastSeg = useRef(null);

  const [started, setStarted] = useState(false);

  // === Init Human ===
  const initHuman = useCallback(async () => {
    const config = {
      backend: "webgl",
      async: true,
      warmup: "face",
      modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/",
      body: { enabled: false },
      face: { enabled: false },
      hand: { enabled: false },
      object: { enabled: false },
      gesture: { enabled: false },
      segmentation: {
        enabled: true,
        modelPath: "selfie.json", // ✅ fastest GPU model
        return: "mask",
        smooth: true,
        threshold: 0.01,
      },
      filter: { enabled: false },
    };

    const human = new Human(config);
    await human.load();
    humanRef.current = human;
    console.log("✅ Human models loaded");
  }, []);

  // === Start Camera ===
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
          offscreenMask.current.width = vid.videoWidth;
          offscreenMask.current.height = vid.videoHeight;
        }
        vid.play();
        onReady();
      };
    } catch (err) {
      console.error("Camera access error:", err);
    }
  }, []);

  // === Temporal blend helper ===
  const temporalBlend = (prev, curr) => {
    if (!prev) return curr;
    const blended = new Uint8ClampedArray(curr.length);
    for (let i = 3; i < curr.length; i += 4) {
      const pa = prev[i];
      const ca = curr[i];
      const a = pa * 0.8 + ca * 0.2; // EMA
      blended[i] = a;
      blended[i - 3] = 0; // R
      blended[i - 2] = 200;   // G
      blended[i - 1] = 200;   // B
    }
    return blended;
  };

  // === Segmentation loop ===
  const segmentLoop = useCallback(async () => {
    const vid = videoRef.current;
    const human = humanRef.current;
    if (!vid || !human) return;

    const segTensor = await human.segmentation(vid);
    if (segTensor) {
      const [height, width] = segTensor.shape; // height first
      const rgba = await segTensor.data();
      const curr = new Uint8ClampedArray(rgba);

      // Threshold and colorize red
      for (let i = 0; i < curr.length; i += 4) {
        const alpha = curr[i + 3];
        curr[i] = 0;
        curr[i + 1] = 200;
        curr[i + 2] = 200;
        curr[i + 3] = alpha > 100 ? alpha : 0; // hard cutoff
      }

      // Temporal blend alpha
      const blended = temporalBlend(lastSeg.current, curr);
      lastSeg.current = blended;

      // Draw to offscreen canvas
      const offCtx = offscreenMask.current.getContext("2d");
      const imageData = new ImageData(blended, width, height);
      offCtx.putImageData(imageData, 0, 0);

      segTensor.dispose();
    }

    setTimeout(segmentLoop, 100); // run ~10FPS
  }, []);

  // === Draw video + blurred mask overlay ===
  const drawLoop = useCallback(() => {
    const vid = videoRef.current;
    const ctx = ctxRef.current;
    if (!vid || !ctx) return;

    ctx.drawImage(vid, 0, 0, ctx.canvas.width, ctx.canvas.height);

    // Draw smoothed mask
    if (offscreenMask.current) {
      ctx.save();
      ctx.globalAlpha = 1;

      // GPU-accelerated blur + contrast
      ctx.filter = "blur(2px) contrast(150%) brightness(120%)";
      ctx.drawImage(offscreenMask.current, 0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.filter = "none";

      ctx.restore();
    }

    vid.requestVideoFrameCallback(drawLoop);
  }, []);

  // === Lifecycle ===
  useEffect(() => {
    let active = true;

    const start = async () => {
      await initHuman();
      if (!active) return;
      await startCamera(() => {
        setStarted(true);
        videoRef.current.requestVideoFrameCallback(drawLoop);
        segmentLoop();
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
  }, [initHuman, startCamera, drawLoop, segmentLoop]);

  return (
    <div className="app">
      <video className="video" ref={videoRef} playsInline muted />
      <canvas className="canvas" ref={canvasRef} />
      {!started && <div className="loading">Loading models and camera...</div>}
    </div>
  );
};

export default App;
