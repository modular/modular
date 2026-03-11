const { mojoRedirects } = require('./mojo/redirects');

const redirects = [
  {
    from: '/max',
    to: '/',
  },
  {
    from: '/magic',
    to: '/pixi',
  },
  {
    from: '/magic/commands',
    to: '/pixi',
  },
  {
    from: '/magic/changelog',
    to: '/pixi',
  },
  {
    from: '/magic/faq',
    to: '/pixi',
  },
  {
    from: '/magic/conda',
    to: '/max/packages',
  },
  {
    from: '/magic/pip',
    to: '/max/packages',
  },
  {
    from: '/max/tutorials/magic/',
    to: '/pixi',
  },
  {
    from: '/max/max-cli',
    to: '/max/cli/',
  },
  {
    from: '/max/issues',
    to: 'https://github.com/modular/modular/issues',
  },
  {
    from: '/max/api/mojo-decorators/compiler-register',
    to: '/mojo/manual/decorators/compiler-register',
  },
  {
    from: '/max/engine',
    to: '/max/intro',
  },
  {
    from: '/max/install',
    to: '/max/get-started',
  },
  {
    from: '/max/graph',
    to: '/max/model-formats',
  },
  {
    from: '/max/graph/get-started',
    to: '/max/develop/get-started-with-max-graph-in-python',
  },
  {
    from: '/max/create-project',
    to: '/max/packages',
  },
  {
    from: '/max/tutorials/deploy-pytorch-llm',
    to: '/max/deploy/local-to-cloud',
  },
  {
    from: '/max/tutorials/max-serve-local-to-cloud',
    to: '/max/deploy/local-to-cloud',
  },
  {
    from: '/max/tutorials/start-a-chat-endpoint',
    to: '/max/inference/text-to-text',
  },
  {
    from: '/max/tutorials/run-embeddings-with-max-serve',
    to: '/max/inference/embeddings',
  },
  {
    from: '/max/tutorials/deploy-llama-vision',
    to: '/max/inference/image-to-text',
  },
  {
    from: '/max/deploy',
    to: '/mammoth/',
  },
  {
    from: '/max/serve',
    to: '/max/deploy/local-to-cloud',
  },
  {
    from: '/max/tutorials/deploy-max-serve-on-kubernetes',
    to: '/mammoth/',
  },
  {
    from: '/max/tutorials/benchmark-max-serve',
    to: '/max/deploy/benchmark',
  },
  {
    from: '/max/tutorials/',
    to: '/max/intro',
  },
  {
    from: '/max/tutorials/serve-custom-model-architectures',
    to: '/max/develop/serve-custom-model-architectures',
  },
  {
    from: '/max/tutorials/max-pipeline-bring-your-own-model',
    to: '/max/develop/max-pipeline-bring-your-own-model',
  },
  {
    from: '/max/tutorials/get-started-with-max-graph-in-python',
    to: '/max/develop/get-started-with-max-graph-in-python',
  },
  {
    from: '/max/tutorials/build-an-mlp-block',
    to: '/max/develop/build-an-mlp-block',
  },
  {
    from: '/max/tutorials/build-custom-ops',
    to: '/max/develop/build-custom-ops',
  },
  {
    from: '/max/tutorials/custom-ops-matmul',
    to: '/max/develop/custom-ops-matmul',
  },
  {
    from: '/max/tutorials/custom-kernels-pytorch',
    to: '/max/develop/custom-kernels-pytorch',
  },
  ...mojoRedirects,
];

module.exports = {
  redirects,
};
