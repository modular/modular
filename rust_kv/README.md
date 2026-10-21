# rust_kv

Rust KV cache components for MAX Serve. All of the crates here are open source
(Apache 2.0 with LLVM Exceptions) and are built from source in the public
repository as well as in the monorepo.

- `kv-core`: host-side KV cache bookkeeping with no GPU, async runtime, or
  Python dependency. Holds the prefix-hit rules and block hashing, the Jenga
  geometry planner, and the Jenga block pool. Mach's `engine-kv` depends on it
  by path.
- `kv-core-pyo3`: the pyo3 bindings of kv-core, which ship in the `max` wheel
  as the compiled module `max._kv_core` (built by
  `//max/python/max:_kv_core.bindings`, the way `max._core` is). Its public
  surface is the hand-written stub `max/python/max/_kv_core.pyi`, which must
  change with the bindings.
- `kv-tier-connector`: the Rust tiered KV connector (host and disk tiers): the
  disk tier, copy engine, lanes, and connectors on top of `kv-core`'s block
  pool. It `dlopen`s the CUDA or HIP driver at runtime, so it builds without a
  GPU toolkit.
- `max-profiler`: safe, dependency-free bindings to the MAX tool-annotation
  shim (`libmax_profiler_shim.so`), which emits NVTX or rocTX ranges for nsys
  and rocprof when `MODULAR_ENABLE_PROFILING` is set. The tiered connector uses
  it for its trace ranges, and Mach's `engine-core` depends on it by path. It
  moved here from `mach/crates/max-profiler` so the connector has no
  dependency on closed-source code.
- `kv-tier-connector-pyo3`: the pyo3 bindings that expose the tiered connector
  to Python as the `kv_tier_connector` module. The private MAX Serve image
  bundles it as the `kv-tier-connector` wheel, whose packaging lives in
  `kv-tier-connector-pyo3/internal` and is not part of the open-source export.

The directory name is deliberately generic: more of the KV cache manager may
move from Python to Rust here over time, not just the connectors.

TODO(MXSERV-705): rename this directory, since it now holds more than KV
crates (`max-profiler` is general-purpose).

The crates used to live under `dkv/`, but they are not part of the dKV
service: merges there were triggering dKV image builds and staging releases, so
the workspace was split out with its own CI (`.github/workflows/rustKv.yaml`)
that builds, lints, and tests without any deploy step.

Nothing here may depend on a crate outside the open-source export (`mach`,
`dkv`, ...).

## Building

```bash
# Cargo (fast local iteration)
cd rust_kv
cargo build --workspace
cargo test --workspace
cargo clippy --workspace

# Bazel, from the repository root
cd ..
./bazelw build //rust_kv/...
./bazelw test //rust_kv/...
./bazelw test //max/tests/integration/kv_cache/internal/kv_core:test_kv_core_binding
```

Third-party crates for the Bazel build come from the `@rust_kv_crates`
repository, generated from this workspace's `Cargo.toml`/`Cargo.lock` (see
`MODULE.bazel` and `oss/modular/oss.MODULE.bazel`). The pyo3 crates are the one
exception: `modular_pyo3_extension` injects pyo3 through the monorepo-wide
`rust_pyo3_toolchain`, so their Bazel targets must not also depend on pyo3
from `@rust_kv_crates`. In the monorepo that toolchain is bound to
`@crates//:pyo3` (the dkv universe), so keep this workspace's pyo3 pin on the
same minor version as `dkv/`'s; the open-source build binds it to
`@rust_kv_crates//:pyo3`.
