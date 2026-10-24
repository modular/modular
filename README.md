<!-- rumdl-disable-file MD033 MD041 MD075 -->

<div align="center">
    <img src="https://modular-assets.s3.us-east-1.amazonaws.com/images/modular-banner-github.png">

[About Modular] | [MAX docs] | [Mojo docs] | [Contributing]

</div>

---

# Modular Platform

This repo hosts open-source components of the Modular Platform,
a unified platform for AI development and deployment,
including the **MAX Framework**🧑‍🚀 and the **Mojo Language**🔥.

## Get started

Want to use MAX and Mojo in your own projects? Check out the
[MAX get started guide](https://max.modular.com/get-started) and the
[Mojo get started guide](https://mojolang.org/docs/manual/quickstart/).

Want to experiment with functionality in MAX or Mojo? Work from the
repository:

- Run `mojo` on a file:

  ```bash
  ./bazelw run //Mojo:mojo -- Mojo/examples/hello.mojo
  ```

- Run `max` to generate text with a model (requires a
  [compatible GPU](https://max.modular.com/packages#gpu-compatibility)):

  ```bash
  ./bazelw run //max/python/max/_entrypoints:pipelines -- generate \
    --model Qwen/Qwen3.5-0.8B --max-batch-size 1 --max-length 10000 \
    --prompt "The sky is blue because"
  ```

## About the repo

We're actively open-sourcing more of the Modular Platform over time.

The main components for which source is available today include:

- Mojo compiler: [/Mojo](/Mojo) (since August 2026)
- Mojo standard library: [/Mojo/stdlib](/Mojo/stdlib) (since March
  2024)
- MAX GPU programming abstractions: [/max/mojo/max/gpu](/max/mojo/max/gpu)
  (GPU programming in Mojo, since May 2025)
- MAX accelerator library: [/max/kernels](/max/kernels) (Mojo compute kernels,
  since May 2025)
- MAX graph compiler: [/GraphCompiler](/GraphCompiler)
- MAX runtime: [/MLRT](/MLRT)
- MAX driver: [/max/driver](/max/driver)
- MAX AI framework: [/max/python/max](/max/python/max) (model authoring in
  Python, since November 2025)
- MAX model pipelines: [/max/python/max/pipelines](/max/python/max/pipelines)
  (models built on the framework, since May 2025)
- MAX inference server: [/max/python/max/serve](/max/python/max/serve)
  (OpenAI-compatible endpoint, since May 2025)
- Code examples: [/max/examples](/max/examples) and
  [/Mojo/examples](/Mojo/examples)

## Contribute

We accept contributions to both Mojo
([Mojo contributor guide](https://mojolang.org/community/contributing/)) and MAX
([MAX contributor guide](/max/docs/contributing/README.md)). For information
about what changes we accept in different areas, see the
[Mojo contribution areas](https://mojolang.org/community/contributing/contribution-areas/)
and [MAX contribution areas](/max/docs/contributing/contribution-areas.md).

We also welcome your bug reports. If you have a bug, please [file an issue
here](https://github.com/modular/modular/issues/new/choose).

## License

Mojo and its contributions are licensed under the Apache License v2.0
with LLVM Exceptions. See [Licenses/APACHE_LICENSE](Licenses/APACHE_LICENSE).

Portions of the MAX source code and its binary distribution are licensed
under the [Licenses/MAX_LICENSE](Licenses/MAX_LICENSE), with the
remainder of the source under the Apache License v2.0 with LLVM
Exceptions. See [Licenses/APACHE_LICENSE](Licenses/APACHE_LICENSE).

### Third party licenses

You are entirely responsible for checking and validating the licenses of third
parties (for example, Hugging Face) for related software and libraries that are
downloaded.

## Community & Events

There are a lot of great ways to get involved! Check out the
[Modular community page][community] for details.

## Thanks to our contributors

<a href="https://github.com/modular/modular/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=modular/modular" />
</a>

<!-- Link references -->

<!-- Header navigation links -->
[About Modular]: https://www.modular.com/
[MAX docs]: https://max.modular.com/
[Contributing]: ./CONTRIBUTING.md
[Mojo docs]: https://mojolang.org/docs/

<!-- Community & Events links -->
[community]: https://www.modular.com/open-source/community
