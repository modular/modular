# MAX pipelines

This is our collection of end-to-end model pipelines that demonstrate the power
of the MAX framework for accelerating common AI workloads. Each of the
supported pipelines can be served via an [OpenAI-compatible
endpoint](https://docs.modular.com/max/api/serve).

As described in our [model support
doc](https://docs.modular.com/max/model-formats), MAX uses the graphs in the
`architectures/` directory here to reconstruct common models from Hugging Face
as performance-optimized MAX graphs. MAX can also serve most PyTorch-based
large language models that are present on Hugging Face, although not at the
same performance as native MAX graph versions.

See [https://builds.modular.com/](https://builds.modular.com/) to discover
hundreds of GenAI models that are supported by these pipelines.
