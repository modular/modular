<div align="center">
    <img src="https://modular-assets.s3.amazonaws.com/images/mojo_github_logo_bg.png">

  [Website][Mojo] | [Getting Started] | [API Documentation] | [Contributing] | [Changelog]
</div>

[Mojo]: https://www.modular.com/mojo/
[Getting Started]: https://docs.modular.com/mojo/manual/get-started/
[API Documentation]: https://docs.modular.com/mojo/lib
[Contributing]: ./CONTRIBUTING.md
[Changelog]: ./docs/changelog.md

# Welcome to Mojo 🔥

Mojo is a new programming language that bridges the gap between research
and production by combining Python syntax and ecosystem with systems
programming and metaprogramming features. Mojo is still young, but it is
designed to write blazing-fast code for CPUs, GPUs, and more as part of
the [MAX Platform](https://www.modular.com/max).

This repo includes source code for:

- Mojo examples
- Mojo documentation hosted at [modular.com](https://docs.modular.com/mojo/)
- The [Mojo standard library](https://docs.modular.com/mojo/lib)

This repo has two primary branches:

- The [`stable`](https://github.com/modular/modular/tree/stable) branch, which
is in sync with the last stable released version of Mojo. Use the examples
here if you’re using a [release build of Mojo](#latest-released).

- The [`main`](https://github.com/modular/modular/tree/main) branch, which
is in sync with the Mojo nightly build and subject to breakage. Use this branch
for [contributions](./CONTRIBUTING.md), or if you're using the latest
[nightly build of Mojo](#latest-nightly).

To learn more about Mojo, see the
[Mojo Manual](https://docs.modular.com/mojo/manual/).

## Installing Mojo

### Latest Released

To install the last released build of Mojo, follow the guide to
[Get started with Mojo](https://docs.modular.com/mojo/manual/get-started).

### Latest Nightly

The nightly Mojo builds are subject to breakage and provide an inside
view of how the development of Mojo is progressing.  Use at your own risk
and be patient!

To get nightly builds, see the same instructions to [Get started with
Mojo](https://docs.modular.com/mojo/manual/get-started), but when you create
your project, instead use the following `magic init` command to set the
conda package channel to `max-nightly`:

```bash
magic init hello-world-nightly --format mojoproject \
  -c conda-forge -c https://conda.modular.com/max-nightly
```

Or, if you're [using conda](https://docs.modular.com/magic/conda), add the
`https://conda.modular.com/max-nightly/` channel to your `environment.yaml`
file. For example:

```yaml
[project]
name = "Mojo nightly example"
channels = ["conda-forge", "https://conda.modular.com/max-nightly/"]
platforms = ["osx-arm64", "linux-aarch64", "linux-64"]

[dependencies]
max = "*"
```

When you clone this repo, you'll be on the `main` branch by default,
which includes code matching the latest nightly build:

```bash
git clone https://github.com/modular/modular.git
```

If you want to instead see the source from the most recent stable
release, then you can switch to the `stable` branch.

## Contributing

When you want to report issues or request features, [please create a GitHub
issue here](https://github.com/modular/modular/issues).
See [here](./CONTRIBUTING.md) for guidelines on filing good bugs.

We welcome contributions to this repo for mojo on the
[`main`](https://github.com/modular/modular/tree/main)
branch. If you’d like to contribute to Mojo, please first read our [Contributor
Guide](https://github.com/modular/modular/blob/main/mojo/CONTRIBUTING.md).

For more general questions or to chat with other Mojo developers, check out our
[Discord](https://discord.gg/modular).

## License

This repository and its contributions are licensed under the Apache License v2.0
with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)).
MAX and Mojo usage and distribution are licensed under the
[MAX & Mojo Community License](https://www.modular.com/legal/max-mojo-license).

## Thanks to our contributors

<a href="https://github.com/modular/modular/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=modular/modular" />
</a>
