
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Forward frontend API requests to the local FastAPI server.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'https://phoenixai-0kvn.onrender.com',
        changeOrigin: true,
      },
    },
  },
})
