import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  // Caminhos relativos mantêm JS e CSS dentro do prefixo /proxy/<porta>/ do Colab.
  base: './',
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL('../../dist/ui', import.meta.url)),
    emptyOutDir: true
  },
  server: { host: '127.0.0.1', port: 5173 }
});
