import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Standalone tester app. Calls the existing backend on :8000 via the dev-server
// proxy so the browser sees same-origin requests and CORS never enters the picture.
// The backend itself is left untouched.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
