# Contribution areas

We plan to gradually expand the types of contributions we accept. This gives us
time to mature our contribution processes, and it lets the Modular team grow and
adapt to the increased input and the review burden that comes with it.

Today every area accepts *bug fixes* only. A bug fix corrects behavior that is
clearly not intended, has a user-observable effect, and comes with a test or
benchmark that reproduces the issue. We'll open each area to larger changes
over time as we set up a process with its team. Where an area accepts larger
changes, open an issue first; the owning team marks it `accepted` once it's
ready for a PR.

Many parts of the codebase are under active development. Contributions to
these components may receive closer review or be superseded by internal work,
so opening an issue to discuss first is strongly recommended.

Before you start work, check that the part of MAX you want to improve is
covered here, then follow the [contribution process](contribution-process.md).
Keep each pull request inside a single area. See
[keep pull requests small](issue-pr-etiquette.md#keep-pull-requests-small).

## Driver

The driver is the device and memory layer underneath every MAX graph: it
enumerates accelerators, allocates host and device buffers, moves data between
them, and launches work on device queues. The Python API lives in
[`max/python/max/driver`](/max/python/max/driver) and the C++ sources
underneath it in [`max/driver`](/max/driver). We accept bug fixes only.

To find a place to start, see the
[open driver issues](https://github.com/modular/modular/issues?q=is%3Aissue+state%3Aopen+label%3Adriver).

## GPU programming

The `max.gpu` package in [`max/mojo/max/gpu`](/max/mojo/max/gpu) is the Mojo
API for writing device code directly, and
[`max/kernels/src/layout`](/max/kernels/src/layout) holds the tensor layouts
kernels are written against. We accept bug fixes, intrinsics or primitives
that are missing for hardware we already support, and documentation. Support
for a new GPU architecture starts in the Mojo standard library; see [adding a
new GPU target](/Mojo/docs/contributing/stdlib/adding-gpu-targets.md).

To find a place to start, see the
[open GPU issues](https://github.com/modular/modular/issues?q=is%3Aissue+state%3Aopen+label%3AGPU).

## Kernels

The MAX accelerator library in [`max/kernels`](/max/kernels) holds the compute
kernels that MAX graphs execute. We accept bug fixes only. A fix needs a test
that checks results against a reference, and a fix for a performance
regression needs a [`kbench`](/max/docs/kernel-benchmarking.md) result.

To find a place to start, see the
[open kernel issues](https://github.com/modular/modular/issues?q=is%3Aissue+state%3Aopen+label%3Amax+label%3Aperformance).

## Graph compiler

The graph compiler in [`GraphCompiler`](/GraphCompiler) lowers a MAX graph to
executable code: it fuses operations, selects kernels, and compiles the result
for the target device. The runtime that executes the result lives in
[`MLRT`](/MLRT), and the Mojo-side registration of built-in kernels and the
extensibility API for custom operations in
[`max/kernels/src/graph_compiler`](/max/kernels/src/graph_compiler). We accept
bug fixes only, with a minimal graph that reproduces the problem.

To find a place to start, see the
[open graph compiler issues](https://github.com/modular/modular/issues?q=is%3Aissue+state%3Aopen+label%3Amax+label%3Agraph-api).

## Framework

The framework is the Python API for building and running models:
[`max/python/max/graph`](/max/python/max/graph),
[`max/python/max/nn`](/max/python/max/nn),
[`max/python/max/engine`](/max/python/max/engine), and the APIs under
development in [`max/python/max/experimental`](/max/python/max/experimental).
We accept bug fixes only.

To find a place to start, see the
[open framework issues](https://github.com/modular/modular/issues?q=is%3Aissue+state%3Aopen+label%3A%22MAX%2FFramework%22).

## Models

The model library in
[`max/python/max/pipelines/architectures`](/max/python/max/pipelines/architectures)
holds the architectures that `max serve` runs natively. We accept bug fixes to
existing architectures, with a logit or accuracy check showing the fix. We're
not currently accepting new models, and plan to open this up again over time.

To find a place to start, see the
[open model issues](https://github.com/modular/modular/issues?q=is%3Aissue+state%3Aopen+label%3Aext-model-contrib).

## Serve

The inference server in [`max/python/max/serve`](/max/python/max/serve)
exposes OpenAI-compatible endpoints and owns request routing, scheduling, and
KV cache management. We accept bug fixes only.

To find a place to start, see the
[open serving issues](https://github.com/modular/modular/issues?q=is%3Aissue+state%3Aopen+label%3Amax+label%3Aruntime).

## Everything else

| Path                       | Description             | Status                  |
|----------------------------|-------------------------|-------------------------|
| `max/docs`                 | Developer documentation | Contributions welcome   |
| `max/examples`             | Code examples           | Contributions welcome   |
| `max/python/max/benchmark` | Benchmarking scripts    | Contributions welcome   |
| `max/python/docs`          | Python API docs         | Automatically generated |
| `max/include/max/c`        | Closed source bindings  | Not accepting changes   |

## Changes we avoid

- Code without tests.
- Changes that break existing APIs, models, or implicit behavior semantics.
- Changes that add dependencies or support for esoteric platforms.
- Broad formatting or refactoring changes.
- Changes where contributors aren't responsive to review comments.
