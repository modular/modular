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
    from: '/max/engine',
    to: '/max/intro',
  },
  {
    from: '/max/install',
    to: '/max/get-started',
  },
  {
    from: '/max/model-formats',
    to: '/max/models',
  },
  {
    from: '/max/graph',
    to: '/max/develop/modules/',
  },
  {
    from: '/max/graph/get-started',
    to: '/max/develop/modules/',
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
    to: '/max/deploy/cloud',
  },
  {
    from: '/max/serve',
    to: '/max/deploy/local-to-cloud',
  },
  {
    from: '/max/tutorials/deploy-max-serve-on-kubernetes',
    to: '/max/deploy/cloud',
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
    to: '/max/develop/modules',
  },
  {
    from: '/max/develop/get-started-with-max-graph-in-python',
    to: '/max/develop/modules',
  },
  {
    from: '/max/tutorials/build-an-mlp-block',
    to: '/max/develop/modules',
  },
  {
    from: '/max/develop/build-an-mlp-block',
    to: '/max/develop/modules',
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
    from: '/mammoth/disaggregated-inference',
    to: '/glossary/ai/disaggregated-inference',
  },
  {
    from: '/mammoth/orchestrator',
    to: '/glossary/ai/inference-routing',
  },
  {
    from: '/mammoth',
    to: '/max/deploy/cloud',
  },
  {
    from: '/glossary/ai/prefill',
    to: '/glossary/ai/context-encoding',
  },
  {
    from: '/glossary/ai/self-attention',
    to: '/glossary/ai/attention',
  },
  {
    from: '/max/api/',
    to: '/max/api/python/',
  },
  {
    from: '/max/api/serve/',
    to: '/max/rest-api/',
  },
];

module.exports = {
  redirects,
};
