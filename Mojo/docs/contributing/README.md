# Mojo contributor guide

Welcome to the Mojo community! 🔥 We're very excited that you're interested in
contributing to the project.

The following procedures and guidelines aim to create an environment where open
source contributors and the teams at Modular can work together effectively
toward the continued improvement of Mojo.

## Before you start

1. Check the [contribution areas](./contribution-areas.md) to
   see whether the part of the codebase you want to improve is accepting
   contributions, and which kinds of change it accepts.

2. For any non-trivial change, [open an
   issue](https://github.com/modular/modular/issues) to discuss it before you
   open a pull request. For a significant change, start with the
   [proposal process](./proposal-process.md) instead.

3. Read our [issue and PR etiquette](./issue-pr-etiquette.md).
   It sets out what we expect from you when engaging with the
   [modular/modular](https://github.com/modular/modular) repository, including
   our rules on AI-assisted contributions.

4. Read the [Code of Conduct on
GitHub](https://github.com/modular/modular/blob/main/CODE_OF_CONDUCT.md).

> [!NOTE]
> We limit new contributors to two concurrent open pull requests.

## How a contribution works

The [contribution process](./contribution-process.md) takes you
from signaling your intent on an issue through to your change shipping in a
nightly.

For the mechanics of forking, branching, and opening a pull request against
this repository, see the [Modular contributor
guide on GitHub](https://github.com/modular/modular/blob/main/CONTRIBUTING.md).

## Developer guides

- [Standard library development](./stdlib/stdlib-development.md): Set up your
  environment, build the library, and run tests.
- [Standard library code style](./stdlib/stdlib-code-style.md):
  Conventions for writing standard library code.
- [Mojo docstring style guide](./stdlib/docstring-style-guide.md): How to write
  API documentation in Mojo.
- [Adding a new GPU target](./stdlib/adding-gpu-targets.md): How to
  extend `std/_gpu/host/info.mojo` with a new GPU architecture, covering the
  MLIR target configuration and the `data_layout` string format.
- [Compiler contributor docs](./compiler/README.md): Where to
  find the documentation you need to work on the compiler.

If you're reading this page on GitHub, you might find it easier to explore all
contributing docs online at
[mojolang.org/community/contributing/](https://mojolang.org/community/contributing/).

## Our priorities

- Our [vision document](https://mojolang.org/docs/vision) describes the guiding
  principles behind our efforts.
- Our [roadmap](https://mojolang.org/docs/roadmap/) identifies concrete short-,
  medium-, and longer-term development goals.

Thank you for your contributions! ❤️
