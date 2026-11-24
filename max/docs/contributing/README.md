# MAX contributor handbook

Welcome to the MAX community! 🧑‍🚀 We're very excited that you're interested in
contributing to the project.

The following procedures and guidelines aim to create an environment where open
source contributors and the teams at Modular can work together effectively
toward the continued improvement of MAX.

There are many ways to contribute, from joining our
[forum](https://forum.modular.com/) or
[Discord community](https://www.discord.gg/modular), to filing bugs, to
contributing documentation, examples, or code.

## Before you start

1. Check the [contribution areas](./contribution-areas.md) to see which kinds
   of change the part of MAX you want to improve accepts. Every area currently
   accepts bug fixes only.

2. Where an area accepts larger changes, [open an
   issue](https://github.com/modular/modular/issues) to discuss the change
   before you open a pull request. The owning team marks the issue `accepted`
   once it's ready for a PR.

3. Read our [issue and PR etiquette](./issue-pr-etiquette.md). It sets out
   what we expect from you when engaging with the
   [modular/modular](https://github.com/modular/modular) repository, including
   our rules on AI-assisted contributions and on keeping pull requests small.

4. Read the [Code of Conduct on
   GitHub](https://github.com/modular/modular/blob/main/CODE_OF_CONDUCT.md).

> [!NOTE]
> We limit new contributors to two concurrent open pull requests.

## How a contribution works

The [contribution process](./contribution-process.md) takes you from signaling
your intent on an issue through to your change shipping in a nightly.

For the mechanics of forking, branching, and opening a pull request against
this repository, see the [Modular contributor guide on
GitHub](https://github.com/modular/modular/blob/main/CONTRIBUTING.md).

## Developer guides

- [MAX framework developer guide](/max/docs/development.md): Set up your
  environment, build MAX, and run the minimal test matrix. Also covers local
  test prerequisites such as Hugging Face auth, model downloads, and GPU-only
  targets.
- [Debug MAX model accuracy](/max/docs/accuracy-debugging.md): Find where a
  MAX pipeline diverges from its PyTorch reference.
- [Kernel benchmarking with `kbench`](/max/docs/kernel-benchmarking.md) and
  [kernel profiling with Nsight Compute](/max/docs/kernel-profiling.md): Show
  that a kernel fix doesn't regress performance.
- [Benchmarking a MAX endpoint](/max/docs/max-benchmarking.md): Measure
  throughput and latency of a serving change.
- [Mojo docstring style guide](/Mojo/docs/contributing/stdlib/docstring-style-guide.md):
  How to document Mojo APIs. Python docstrings use RST with Google-style
  sections; match the module you're editing.

Thank you for your contributions! ❤️
