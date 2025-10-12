import React, { useState } from 'react';
import WebcamFeed from './components/webcamfeed';

const App = () => {
  const [started, setStarted] = useState(false);

  return (
    <div className="app">
      <WebcamFeed started={started} setStarted={setStarted} />
    </div>
  );
};

export default App;
