# Eager-interpreter GC-sweep warm store

The eager interpreter compiles a graph-compiler `Model` for each op (matmul,
unary) on first dispatch. On a cold cache that first dispatch lowers and JIT-
compiles a large slice of the kernel library, thousands of kernels, minutes.
This package pays that cost once, at build time, as a
[`MefStore`](../../../python/max/engine/_precompiled_mefs.py) a test loads
instead of compiling.

## How it works

- Build actions, one per (op family, device class), each write a `MefStore`
  directory holding one artifact per (family, device slot):
  `matmul_cpu.mef`, `matmul_slot_0.mef`, and so on. Splitting by family keeps
  each action under the action timeout and lets them run in parallel; splitting
  by device class lets the CPU sweep, which is identical on every x86 lane, run
  once and be shared from the remote cache.
  - `warm_cpu.py` warms the CPU slot. It takes only `--cpu-target`, so its
    action is the same on every lane.
  - `warm_accelerators.py` presents `WARM_DEVICE_COUNT` virtual accelerators of
    the lane's arch (`--target`) and warms their slots, with no physical GPU.
  - `warm_lib.py` is their shared compile-and-export loop.
  - `interp_cache.bzl` runs a producer as a build action whose output is the
    store directory.
- `:warm_cpu` and `:warm_gpu` are filegroups of a lane's fragments. Nothing
  merges them: `MefStore` unions the directories it is given.
- `eager_py_test.bzl` (`modular_eager_py_test`) gives a test its lane's
  fragments, and `eager_warm_plugin.py` names them in `MAX_EAGER_WARM_STORE`.
  Every family then loads its artifacts from the store
  (`gc_compile.warm_store`), at import under `MAX_EAGER_OP_PRECOMPILE=1`.
- Adoption is all-or-nothing. A store missing an artifact a family needs raises
  `MissingArtifactError` instead of compiling: a box with more accelerators than
  `WARM_DEVICE_COUNT` fails loudly. Which warm a lane gets is decided by the
  lane `select`, so there is no runtime check of host or GPU arch.
- Only B200 and MI355 have a GPU warm. A CPU lane gets the CPU-only warm, and
  any other GPU lane (H100, Apple, a local box) gets none, so the interpreter
  compiles its op families lazily there.
- A family must exclude from `sweep_devices()` any device it builds nothing
  for; the producer fails if it doesn't, so producer and consumer always agree
  on which slots exist.

[Eager execution in CI](../../../../docs/internal/EagerInCI.md) explains
these settings and the defaults they override. `modular_eager_py_test` also
adds the `_interpreter_ops` dep and sets
`MAX_EAGER_EXECUTOR=interpreter`, so a graph the interpreter refuses fails its
test instead of compiling on the GPU worker. A test that needs such a graph
says so with `@pytest.mark.compile_on_gpu(reason)`; the plugin then runs that
test with the compile-on-refusal executor. CPU suites may compile, and set
`MAX_EAGER_EXECUTOR=fallback-internal`.

`test_interp_warm_cache` (multi-GPU) and `test_cpu_warm_cache` (GPUs hidden)
check that dispatches use loaded models with no cold recompile.

`max warm-interpreter-cache` and its warm stamp are a separate, same-machine
path for product use; CI does not rely on them.

Run it (Linux only; the eager interpreter tests are macOS-incompatible,
FIXME MOCO-2411):

    bazel test //max/tests/integration/xarch_warm:test_interp_warm_cache

## Caveats

- **x86 GPU builders only.** The sole eager GPU consumer lane is the x86 + B200
  runner, so `warm_accelerators.py` aborts on a non-x86 host rather than emit
  an unexercised artifact. `warm_cpu.py` has no such guard: aarch64 is a valid
  CPU-only target, and the per-platform CPU-target `select()` carries an
  aarch64 entry.
- **Device arch must be `sm_100a`**, not the bare `sm_100`, `ptxas` rejects
  the bare form for architectures that require the accelerated-features suffix.
- **`x86-64-v3` is the floor.** The warm's host-ELF kernels are compiled for
  the `x86-64-v3` target the per-platform CPU `select()` pins, so every x86 box
  that loads them must be at least v3; a v2 box would `SIGILL`. All CI x86
  runners are.
- **`set_virtual_cpu_target` grammar.** Only the descriptor form
  (`triple=<triple>;cpu=<name>`) pins the triple; a bare CPU name (`x86-64-v3`)
  only re-tunes `-mcpu` on the host's own triple.
