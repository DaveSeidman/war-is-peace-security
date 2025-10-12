import React, { useEffect, useRef } from 'react';

const OverlayCanvas = ({ tracks }) => {
  const overlayRef = useRef();

  useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext('2d');
    ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);

    ctx.lineWidth = 8;
    ctx.font = '16px sans-serif';
    ctx.textBaseline = 'bottom';
    ctx.strokeStyle = 'rgba(173, 216, 230, 1.0)';
    ctx.fillStyle = 'rgba(173, 216, 230, 1.0)';

    tracks.forEach(track => {
      const { x, y, w, h, id } = track;
      ctx.strokeRect(x, y, w, h);
      ctx.fillText(id, x, y - 8);
    });

  }, [tracks]);

  return (
    <canvas
      ref={overlayRef}
      width={640}
      height={480}
      style={{ position: 'absolute', top: 0, left: 0 }}
    />
  );
};

export default OverlayCanvas;
