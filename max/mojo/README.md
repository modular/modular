# MAX Mojo package

This directory contains the `max` package for hardware-accelerated programming
in [Mojo](https://mojolang.org/).

The `max` package provides foundational primitives, algorithms, runtime
bindings, and GPU abstractions for building high-performance AI and numerical
workloads.

## Package structure

- [`max`](./max): The `max` Mojo package source code.
  - `gpu`: GPU programming APIs (`max.gpu`).
  - `algorithm`: Parallel algorithms and compute primitives (`max.algorithm`).
  - `benchmark`: Benchmarking utilities (`max.benchmark`).
  - `runtime`: Async runtime APIs (`max.runtime`).
- [`test`](./test): Unit and integration tests for the `max` Mojo package.

## License

Apache License v2.0 with LLVM Exceptions

See the [LICENSE](../../LICENSE) file in the repository root for more details.

## Contributing

Thanks for your interest in contributing to the MAX Mojo package! Please refer
to the [MAX Contributor Guide](../CONTRIBUTING.md) and the repo's
[Contributor Guide](../../CONTRIBUTING.md) for contribution guidelines.

## Support

For any inquiries, bug reports, or feature requests, please [open an
issue](https://github.com/modular/modular/issues) on the GitHub repository.
