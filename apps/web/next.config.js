// @ts-check

 
const { composePlugins, withNx } = require('@nx/next');

/**
 * @type {import('@nx/next/plugins/with-nx').WithNxOptions}
 **/
const nextConfig = {
  nx: {},
  experimental: {
    serverActions: {
      bodySizeLimit: '50mb',
    },
  },
  // 브라우저 → 같은 origin(/api/*) → Next 서버가 FastAPI로 프록시.
  // 외부 클라이언트가 localhost:8000을 직접 못 찾는 문제 해결 (api 포트 비노출 유지).
  async rewrites() {
    const backend = process.env.BACKEND_ORIGIN || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${backend}/:path*`,
      },
    ];
  },
};

const plugins = [withNx];

module.exports = composePlugins(...plugins)(nextConfig);
