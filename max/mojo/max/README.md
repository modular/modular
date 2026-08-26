# `max` Mojo package

This directory contains the root source code for the `max` Mojo package for
hardware-accelerated programming in [Mojo](https://mojolang.org/).

## Modules

- [`gpu`](./gpu): GPU programming APIs (`max.gpu`).
- [`algorithm`](./algorithm): Parallel algorithms and compute primitives
  (`max.algorithm`).
- [`benchmark`](./benchmark): Benchmarking utilities (`max.benchmark`).
- [`runtime`](./runtime): Async runtime APIs (`max.runtime`).

## Usage

In your Mojo code:

```mojo
import max
import max.gpu
import max.algorithm
```

## License

Apache License v2.0 with LLVM Exceptions

See the [LICENSE](../../../LICENSE) file in the repository root for more
details.
