import { defineConfig } from 'vite';

export default defineConfig({
  base: '/', // For custom domain latent.guardabrazo.com
  server: {
    port: 3000, // You can change the port if needed
  },
  build: {
    outDir: 'dist',
  },
  publicDir: 'public',
});
