import React, { useRef, useEffect, useState } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { Pose } from '@mediapipe/pose';
import { matchTracks, resetTrackIds } from '../../utils/';
import OverlayCanvas from '../overlaycanvas';
import testVid from '../../assets/videos/test1b.mp4';
import './index.scss';

// Generate a unique but stable color per ID
const colorForId = (id) => {
  // Hash the ID into a hue (0–360)
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = id.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 100%, 65%)`;
};

const WebcamFeed = ({ started, setStarted }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [pose, setPose] = useState(null);
  const [detector, setDetector] = useState(null);
  const [tracks, setTracks] = useState([]);

  // === Load models ===
  useEffect(() => {
    (async () => {
      const det = await cocoSsd.load();
      setDetector(det);

      const p = new Pose({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      p.setOptions({
        modelComplexity: 0,
        smoothLandmarks: true,
        enableSegmentation: true,
        smoothSegmentation: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      // Wrap onResults to promise-style
      const runPose = (image) =>
        new Promise((resolve) => {
          p.onResults((r) => resolve(r));
          p.send({ image });
        });

      setPose({ instance: p, runPose });
    })();

    // Reset track counter when component mounts
    resetTrackIds();
  }, []);

  // === Camera control ===
  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    setStarted(true);
  };

  const startVideo = async () => {
    if (videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    videoRef.current.src = testVid;
    videoRef.current.loop = true;
    await videoRef.current.play();
    setStarted(true);
  };

  // === Main loop (detect + track + pose) ===
  useEffect(() => {
    if (!pose || !detector || !started) return;

    const v = videoRef.current;
    const ctx = canvasRef.current.getContext('2d');
    let running = true;
    let prevTracks = [];

    const process = async () => {
      if (!running || v.videoWidth === 0) return;

      const detections = await detector.detect(v);
      const people = detections
        .filter((d) => d.class === 'person' && d.score > 0.5)
        .map((d) => ({
          x: d.bbox[0],
          y: d.bbox[1],
          w: d.bbox[2],
          h: d.bbox[3],
        }));

      const tracked = matchTracks(people, prevTracks);
      setTracks(tracked);
      prevTracks = tracked;

      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      for (const t of tracked) {
        const color = colorForId(t.id);

        // === crop and run pose ===
        const off = document.createElement('canvas');
        off.width = t.w;
        off.height = t.h;
        const octx = off.getContext('2d');
        octx.drawImage(v, t.x, t.y, t.w, t.h, 0, 0, t.w, t.h);

        const results = await pose.runPose(off);

        // === draw segmentation mask if available ===
        if (results?.segmentationMask) {
          const maskCanvas = document.createElement('canvas');
          maskCanvas.width = t.w;
          maskCanvas.height = t.h;
          const mctx = maskCanvas.getContext('2d');

          // draw mask as alpha layer
          mctx.drawImage(results.segmentationMask, 0, 0, t.w, t.h);
          const maskData = mctx.getImageData(0, 0, t.w, t.h);
          const d = maskData.data;
          const rgb = color.match(/\d+/g).map(Number); // extract r,g,b from hsl/rgb

          for (let i = 0; i < d.length; i += 4) {
            const alpha = d[i]; // mask intensity (0–255)
            d[i] = rgb[0];      // R
            d[i + 1] = rgb[1];  // G
            d[i + 2] = rgb[2];  // B
            d[i + 3] = alpha * 0.5; // 50% opacity
          }
          mctx.putImageData(maskData, 0, 0);

          // draw the tinted mask back onto main canvas
          ctx.drawImage(maskCanvas, t.x, t.y, t.w, t.h);
        }

        // === outline + label ===
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(t.x, t.y, t.w, t.h);
        ctx.fillStyle = color;
        ctx.font = '16px sans-serif';
        ctx.fillText(`ID ${t.id}`, t.x + 5, Math.max(0, t.y - 20));
      }

      requestAnimationFrame(process);
    };


    process();
    return () => (running = false);
  }, [pose, detector, started]);

  // === Cleanup ===
  useEffect(() => {
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  return (
    <div className="webcamfeed">
      <video
        className="webcamfeed-video"
        ref={videoRef}
        width={640}
        height={480}
        playsInline
        muted
      />
      <canvas
        className="webcamfeed-canvas"
        ref={canvasRef}
        width={640}
        height={480}
      />
      <OverlayCanvas tracks={tracks} />

      <div className="webcamfeed-controls">
        <button onClick={startCamera}>Start Camera</button>
        <button onClick={startVideo}>Start Video</button>
      </div>
    </div>
  );
};

export default WebcamFeed;
