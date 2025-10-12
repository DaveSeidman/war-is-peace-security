import React, { useState, useRef, useEffect } from "react";
import "./index.scss";

export default function Camera({ started, setStarted, takePhoto, setTakePhoto, setOriginalPhoto, setOriginalBlob }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const frameRef = useRef(null);

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    videoRef.current.srcObject = stream;
    setStarted(true);
  };

  useEffect(() => {
    if (!started) return;

    const ctx = canvasRef.current.getContext("2d");

    const draw = () => {
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
      frameRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(frameRef.current);
    };
  }, [started]);

  useEffect(() => {
    if (takePhoto) {
      const canvas = canvasRef.current;

      // Convert to blob for upload (preferred over dataURL for large images)
      canvas.toBlob(async (blob) => {
        if (!blob) return;

        // optional: preview locally
        const localURL = URL.createObjectURL(blob);
        setOriginalPhoto(localURL);

        // Upload to your Node server
        setOriginalBlob(blob);
        setTakePhoto(false);
      }, "image/jpeg", 0.9); // adjust quality if needed
    }
  }, [takePhoto]);

  return (
    <div className="camera">
      <video
        className="camera-video"
        ref={videoRef}
        autoPlay
        playsInline
        muted
      />
      <canvas ref={canvasRef} className="camera-canvas" />

      {!started && (
        <button className="camera-start" onClick={startCamera}>startCamera</button>
      )}
    </div>
  );
}
