import React, { useRef, useEffect, useState } from 'react';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import '@tensorflow/tfjs-backend-webgl';
import { matchTracks } from '../../utils'
import OverlayCanvas from '../overlaycanvas';
import testVid from '../../assets/videos/test1b.mp4';
import './index.scss'

const WebcamFeed = ({ started, setStarted }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [segmenter, setSegmenter] = useState(null);
  const [tracks, setTracks] = useState([]);

  // Load segmentation model once
  useEffect(() => {
    (async () => {
      const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
      const config = {
        runtime: 'mediapipe',
        solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation`
      };
      const loaded = await bodySegmentation.createSegmenter(model, config);
      setSegmenter(loaded);
    })();
  }, []);

  // Start webcam manually
  const handleStartCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setStarted(true);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  // Start test video
  const handleStartVideo = async () => {
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = testVid;
      videoRef.current.loop = true;
      await videoRef.current.play();
      setStarted(true);
    }
  };

  // Main processing loop
  useEffect(() => {
    if (!segmenter || !started) return;
    let running = true;

    const processFrame = async () => {
      if (!running || !videoRef.current) return;
      const v = videoRef.current;
      if (v.videoWidth === 0 || v.videoHeight === 0) {
        requestAnimationFrame(processFrame);
        return;
      }

      const segmentation = await segmenter.segmentPeople(v);
      const updatedTracks = matchTracks(segmentation, tracks);
      setTracks(updatedTracks);

      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      if (segmentation.length > 0) {
        const mask = await bodySegmentation.toBinaryMask(
          segmentation,
          { r: 173, g: 216, b: 230, a: 255 },
          { r: 0, g: 0, b: 0, a: 0 }
        );
        await bodySegmentation.drawMask(
          canvasRef.current,
          v,
          mask,
          0.5, // opacity
          0,   // blur
          false
        );
      }

      requestAnimationFrame(processFrame);
    };

    const onPlay = () => requestAnimationFrame(processFrame);
    videoRef.current.addEventListener('play', onPlay);

    return () => {
      running = false;
      videoRef.current?.removeEventListener('play', onPlay);
    };
  }, [segmenter, started]);

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
      />
      <OverlayCanvas tracks={tracks} />

      {/* Control Buttons */}
      <div className='webcamfeed-controls'>
        <button onClick={handleStartCamera}>Start Camera</button>
        <button onClick={handleStartVideo}>Start Video</button>
      </div>
    </div>
  );
};

export default WebcamFeed;
