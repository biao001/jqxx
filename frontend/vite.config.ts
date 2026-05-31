import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  // 后端地址(供 dev server 反向代理)。默认本机 8000；可用 DMS_BACKEND 覆盖。
  const backend = env.DMS_BACKEND || 'http://127.0.0.1:8000';
  // HTTPS：mode=https 或 DMS_HTTPS=1 时启用（浏览器只在 https/localhost 下才允许摄像头）。
  // 证书放仓库根 .certs/ 下（dms-dev.key / dms-dev.crt）。localhost 下 HTTP 也可用摄像头。
  const certDir = path.resolve(__dirname, '..', '.certs');
  const certKey = path.join(certDir, 'dms-dev.key');
  const certFile = path.join(certDir, 'dms-dev.crt');
  const https =
    mode === 'https' || env.DMS_HTTPS === '1'
      ? {
          key: fs.readFileSync(certKey),
          cert: fs.readFileSync(certFile),
        }
      : undefined;
  return {
    plugins: [react(), tailwindcss()],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      host: '0.0.0.0',
      https,
      // 同源代理:前端只暴露一个端口(3000)，/api、/health、/ws 转发到后端，
      // 局域网其他机器无需访问 8000，规避代理/防火墙导致的"后端未链接"。
      proxy: {
        '/api': { target: backend, changeOrigin: true },
        '/health': { target: backend, changeOrigin: true },
        '/ws': { target: backend, ws: true, changeOrigin: true },
      },
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
