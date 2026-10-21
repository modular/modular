# MAX accelerator library contributor guide

Thank you for your interest in contributing to the MAX accelerator library!
The library follows the [MAX contributor
handbook](/max/docs/contributing/README.md), which sets the policy for every
area of MAX and describes the [contribution
process](/max/docs/contributing/contribution-process.md).

## Changes we accept

Today the library accepts *bug fixes* only. See the [kernels contribution
area](/max/docs/contributing/contribution-areas.md#kernels) for the current
policy. A fix needs a test that checks results against a reference, and a fix
for a performance regression needs a
[`kbench`](/max/docs/kernel-benchmarking.md) result on the hardware it
targets.

We plan to open the library to new kernels and other larger changes over time,
with a proposal process for changes that could affect the performance of the
core kernel library. Until then, if you have a kernel in mind, [open an
issue](https://github.com/modular/modular/issues) describing it so we can
discuss where it fits.

## Developing in the library

Run the tests for the kernel you changed rather than the whole library:

```bash
./bazelw test //max/kernels/test/linalg/...
```

GPU tests live under `max/kernels/test/gpu` and need the hardware they target.
If you can only test on one of the GPUs a change affects, say so in the PR.
For benchmarking and profiling, see [kernel benchmarking with
`kbench`](/max/docs/kernel-benchmarking.md) and [kernel profiling with Nsight
Compute](/max/docs/kernel-profiling.md).

Thank you for your contributions! ❤️
