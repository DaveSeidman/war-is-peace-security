import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/war-is-peace-security/',
  plugins: [react()],
  server: {
    port: 8080,
    host: true,
  },
});
