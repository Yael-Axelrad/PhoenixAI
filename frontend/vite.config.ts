
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Forward frontend API requests to the local FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
