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
};

const plugins = [withNx];

module.exports = composePlugins(...plugins)(nextConfig);
