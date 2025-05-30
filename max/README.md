# MAX framework

MAX is a high-performance inference server that provides
an [OpenAI-compatible endpoint](https://docs.modular.com/max/api/serve) for
large language models (LLMs) and it's a fundamental component of the
[Modular Platform](https://docs.modular.com/max/intro).

This directory includes the source for our Python-based inference server,
Python-based model pipelines (graphs), Python-based neural-net operators
(high-level graph ops), Mojo-based kernel functions (low-level graph
ops for GPUs and CPUs), and more.

## Usage

With just a few commands, you can use MAX to create a local endpoint serving a
large language model (LLM) of your choice, using our CLI tool or Docker
container. Try it now with our [quickstart
guide](https://docs.modular.com/max/get-started).

See [https://builds.modular.com/](https://builds.modular.com/) to discover many
of the models supported by MAX.
