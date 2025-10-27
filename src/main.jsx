import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter, Routes, Route } from 'react-router-dom';
import App1 from './App1';
import App2 from './App2';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <HashRouter>
    <Routes>
      <Route path="/" element={<App1 />} />
      <Route path="/1" element={<App1 />} />
      <Route path="/2" element={<App2 />} />
    </Routes>
  </HashRouter>

);
