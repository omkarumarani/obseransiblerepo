/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    port: 3000,
    // In local dev, proxy API routes to ui-backend running at port 9005.
    // In production the nginx.conf handles the same routing.
    proxy: {
      '/pipeline': { target: 'http://localhost:9005', changeOrigin: true },
      '/scenarios': { target: 'http://localhost:9005', changeOrigin: true },
      '/autonomy':  { target: 'http://localhost:9005', changeOrigin: true },
      '/health':    { target: 'http://localhost:9005', changeOrigin: true },
      '/admin':     { target: 'http://localhost:9005', changeOrigin: true },
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
  },

  // Vitest configuration — runs alongside the Vite build config so
  // you don't need a separate vitest.config.ts.
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
});
