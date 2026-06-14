// vite.config.js
//
// WHY proxy?
// Frontend localhost:5173 pe run karta hai, FastAPI localhost:8000 pe.
// Browser CORS block kar deta hai cross-origin requests.
// Vite proxy = frontend ka server quietly forward karta hai /api/* ko FastAPI pe.
// Production mein Nginx yeh kaam karta hai.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // /analyze → http://localhost:8000/analyze
      "/analyze": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      // /chart/xxx.png → http://localhost:8000/chart/xxx.png
      "/chart": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
