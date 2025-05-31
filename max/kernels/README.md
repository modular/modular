# MAX AI kernels

This directory contains low-level, high-performance compute kernels written in
[Mojo](https://www.modular.com/mojo), designed to serve as building blocks for
numerical, machine learning, and other performance-critical workloads.

Kernels in this directory are written using Mojo's systems programming
capabilities, including fine-grained control over memory layout, parallelism,
and hardware mapping. These implementations prioritize performance and
correctness, and are intended to be used both directly and as primitives in
higher-level libraries.

To evaluate kernel performance on NVIDIA hardware, see [Kernel profiling with
Nsight Compute](docs/profiling.md).

If you're looking for the high-level Python APIs based on these kernels and
used to build MAX graphs, see the [`max/nn/`](../nn) directory.

## Contributing

We will start accepting contributions in early June 2025. See the
[Contributing Guide](./CONTRIBUTING.md) for details.

## License

Apache License v2.0 with LLVM Exceptions

See the license file in the repository for more details.

## Support

For any inquiries, bug reports, or feature requests, please [open an
issue](https://github.com/modular/modular/issues) on the GitHub repository.
