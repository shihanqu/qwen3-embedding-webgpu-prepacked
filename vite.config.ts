import { defineConfig } from 'vite';

const releaseModelProxy = () => ({
  target: 'https://github.com',
  changeOrigin: true,
  followRedirects: true,
  rewrite: (path: string) => path.replace(
    /^\/model-release/,
    '/shihanqu/qwen3-embedding-webgpu/releases/download/model-q4_0-v1',
  ),
});

const isolationHeaders = {
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
};

export default defineConfig({
  publicDir: false,
  server: {
    proxy: {
      '/model-release': releaseModelProxy(),
      '/baseline': {
        target: process.env.LM_STUDIO_URL ?? 'http://127.0.0.1:1234',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/baseline/, ''),
      },
      '/q4-reference': {
        target: process.env.Q4_K_M_REFERENCE_URL ?? 'http://127.0.0.1:1235',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/q4-reference/, ''),
      },
      '/q40-reference': {
        target: process.env.Q4_0_REFERENCE_URL ?? 'http://127.0.0.1:1236',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/q40-reference/, ''),
      },
    },
    headers: isolationHeaders,
  },
  preview: {
    proxy: {
      '/model-release': releaseModelProxy(),
    },
    headers: isolationHeaders,
  },
});
