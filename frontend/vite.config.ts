import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The dev server proxies the API and WebSocket to the FastAPI backend so the
// dashboard runs same-origin in development (no CORS, no absolute URLs).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
});
