# wasm64 external dependency pins

The wasm64-wasip1 bring-up is **experimental**: it targets the wasm64
WASI preview1 ABI, whose process-level details are still pre-standard. It
depends on several external
artifacts. The wasi-libc sysroot (including the compiler-rt builtins and the
KGEN_CompilerRT_* glue archive) is **Bazel-managed**; the WAVM runtime is
still built by hand into `/tmp`. This file records the exact pins and how to
reproduce them.

## wasi-libc wasm64 sysroot (Bazel-managed)

- **Pin:** wasi-libc PR #444 (wasm64 support), commit
  `33983baf671eaa50da2ec0f7c9e9eeceb1f089b1`, sha256
  `d9e5c5ef438e57e68d4a7949f3ac0cbef64046b3235ef437beed2af39be3d214`
  (`@wasi_libc_wasm64` in `bazel/common.MODULE.bazel`).
- **Built by:** `./bazelw build //Mojo/lib/CompilerRT/wasi:wasi_sysroot`
  (the `wasm64_sysroot` rule in `wasm64_sysroot.bzl`, action script
  `build_wasm64_sysroot_action.sh`). The build uses the repo's pinned
  clang-22 (`@clang-macos` / `@clang-linux-*`) and the compiler-rt builtins
  sources from `@llvm-raw`. The action is tagged `no-remote-exec` and also
  needs host `tar`, `make`, and `getconf` (wasi-libc builds in-tree with
  its Makefile); the action script checks for them up front.
- **Location:** `bazel-bin/Mojo/lib/CompilerRT/wasi/wasi_sysroot` — a
  complete `--wasi-sysroot` tree. The same action also installs
  `lib/wasm64-wasip1/libclang_rt.builtins-wasm64.a` (compiler-rt builtins)
  and `lib/wasm64-wasip1/libkgen_compilerrt_wasi.a` into it.
- **Hand-built fallback:** `build_wasi_sysroot.sh` + `build_wasm64_shim.sh`
  into `/tmp` (override: `WASI_SYSROOT`, `WASI_SHIM`); the smoke test falls
  back to these when the Bazel artifact is absent.
- **Consumed as:** `--wasi-sysroot` for `mojo build`.

## WAVM runtime (wasm64 preview1 execution)

- **Pin:** fork `trcrsired/WAVM`, branch `mt-2` (head
  `b17e158616333cb8c4e19a5b7d72fccde070b464`), plus the local patches
  `Mojo/lib/CompilerRT/wasi/wavm-sock-accept.patch` and
  `wavm-unpacked-assert-metadata.patch` (the fork packs pointer members of
  `AssertMetadata` to alignment 1, which the modern macOS linker rejects).
- **Built by:** `build_wavm.sh` (this directory). Not Bazel-managed: WAVM
  links against an unpinned LLVM CMake package, so CI runs the wasm64
  tests under wasmtime instead. Verified recipe on macOS arm64: Homebrew
  `llvm` 23.1.1 (CMake package + monolithic `libLLVM` dylib), system
  clang as compiler (C++17 override — WAVM defaults its standard to 14
  and LLVM 17+ headers need 17). The script applies both patches,
  autodetects the Homebrew/Debian LLVM CMake package, and rewrites the
  CMake link commands to use the shared `libLLVM` so libWAVM stays under
  arm64's +/-128MB branch range.
- **Default location:** `/tmp/WAVM-build/bin/wavm` (override: `WAVM_BIN`).
- **Used for:** running wasm64 WASI commands (`wavm run --abi=wasi --enable
  memory64`, plus `--mount-root` for file-I/O tests).

## Wasmtime (CI runner)

- **Pin:** the python3 `wasmtime` module (system install). Verified
  **45.0.0**; the CI workflow installs `wasmtime==45.0.0` explicitly
  (the hosted runner image does not ship it). Do not blind-upgrade:
  measured on arm64 macOS (interleaved,
  min of 5), 47.0.1 / 48.0.0 / 49.0.0 all regress the relaxed-simd
  fusion this stack relies on — the 8×SIMD throughput shape goes
  95 → 171–190 ms (f32x4 relaxed_madd no longer fused) and scalar f64
  168 → 224 ms. (46.0.0 has no py3.14 wheel.) 45 fuses both f32x4 and
  f64x2 relaxed_madd in clean codegen contexts; see PR_READINESS for
  the code-shape guidance.
- **Used for:** `tools/wasm64/wasmtime-runner.py`, which runs wasm64 WASI
  modules under Wasmtime with a hand-rolled i64 WASI preview1 host
  (Wasmtime's built-in preview1 is i32-only). This is what CI uses for
  `wasm64_smoke_test` (`WASM64_RUNTIME=wasmtime`). The host speaks the
  `_wasm64`-suffixed wasi-libc import names and implements the
  wasm64 (8-byte `size_t`) ABI for prestats, environ, argv, iovecs, and
  filestats.

## compiler-rt builtins (wasm64)

- **Pin:** LLVM compiler-rt builtins subset listed in
  `wasm64_builtins_sources.txt` (this directory), built from `@llvm-raw` by
  the `:wasi_sysroot` action with the repo's clang-22 (hand-built fallback:
  `build_wasm64_builtins.sh`).

## Repo-built inputs (Bazel-managed)

- `bazel-bin/Mojo/tools/mojo/mojo` — native mojo compiler.
- `bazel-bin/Mojo/stdlib/std/std.mojoc` — stdlib package, via
  `./bazelw build --config=build-mojo '@mojo//:std'`.
