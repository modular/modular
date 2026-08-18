# Building the Mojo release with Nix — from source, pure, offline

`flake.nix` (+ this directory) rebuilds the open-source Mojo release
(compiler, standard library, debugger, LSP server and their wheels) from the
sources in this repository, **without downloading or executing a single
prebuilt binary**: Bazel, clang, the sysroot and CPython all come from
nixpkgs' source-built packages, and the whole build runs natively in the Nix
sandbox — no FHS environment, no patchelf, no nix-ld.

```text
nix build .#mojo            # Mojo compiler + stdlib + debugger/LSP
nix build .#mojo-wheels     # the release wheels of the Mojo family
nix build .#release         # wheels + PEP 503 index (offline pip install) + source tarball
nix run   .#mojo -- --version
nix develop                 # nixpkgs Bazel pre-configured: bazel-nix build --config=build-mojo //KGEN:mojo
nix flake check             # compiles, runs and debugs a Mojo program with the result
```

## What is built (and what is deliberately not)

| wheel                                          | contents                                                        | status                |
| ---------------------------------------------- | --------------------------------------------------------------- | --------------------- |
| `mojo_compiler-<v>-py3-none-linux_<arch>.whl`  | `bin/mojo`, `bin/lld`, crashpad handler, runtime `.so`, py pkg  | **built from source** |
| `mojo_compiler_mojo_libs-<v>-py3-none-any.whl` | `lib/mojo/std.mojoc` (standard library)                         | **built from source** |
| `mojo_lldb_libs-<v>-py3-none-linux_<arch>.whl` | `libMojoLLDB.so`, `libMojoJupyter.so`, `liblldb*.so`            | **built from source** |
| `mojo-<v>-py3-none-linux_<arch>.whl`           | `mojo-lldb`, `lldb-server`, `lldb-dap`, `mojo-lsp-server`, REPL | **built from source** ¹ |
| `mblack-<v>-py3-none-any.whl`                  | the Mojo formatter (pure python)                                | **built from source** |
| `max-*`, `max_core-*`, `max_mojo_libs-*`, …    | MAX engine/serving                                              | **not built** ²       |

¹ except `gpu-query`, whose sources are not in the public repository.
² MAX's core (`libmax.so`, the graph compiler, `_core.*.so`, the internal
  kernel packages) is not open source — the public tree itself consumes it
  from prebuilt wheels (`bazel/modular_wheel_repository.bzl`).  Since this
  flake refuses prebuilt binaries, MAX is excluded from the release entirely.
  (The upstream *Mojo* wheels remain pinned in `release-wheels.json` solely
  for `nix build .#release-compare`, which diffs our wheels' file lists
  against upstream's.)

The wheels use a `linux_<arch>` platform tag instead of upstream's
`manylinux_2_34_<arch>`: the binaries are ordinary Nix binaries (interpreter
and rpath point into `/nix/store`), not manylinux ones.  Versions
(`mojo --version`, wheel names, `Requires-Dist`) come from
`bazel/mojo.MODULE.bazel` via `nix/versions.nix` — the release this checkout
tracks.

## How it works

Modular's Bazel build normally downloads: a Bazel-at-commit binary (via the
BuildBuddy CLI), a prebuilt clang 22 + Ubuntu jammy sysroot, prebuilt
llvm-ifs, python-build-standalone, and the nightly Mojo toolchain.  Every one
of those is replaced by a source-built nixpkgs equivalent:

| downloaded by upstream                     | replaced by (all from nixpkgs source builds)                            |
| ------------------------------------------ | ----------------------------------------------------------------------- |
| Bazel (pinned commit binary)               | `pkgs.bazel_9` (9.1.1, bootstrapped from source by nixpkgs)             |
| `@clang-linux-*` (clang 22.1.4 tarball)    | `llvmPackages_22` clang/lld/LLVM tools, laid out like the tarball, described by Modular's own `bazel/public-patches/clang.BUILD` |
| `sysroot-jammy-*` (Ubuntu sysroot tarball) | a sysroot assembled from nixpkgs: glibc (headers incl. kernel headers, crt, libc), gcc (libstdc++ headers/libs, libgcc, crtbegin), ncurses/libedit/libbsd/libmd/zlib for LLDB |
| `@llvm-ifs` (prebuilt)                     | `llvm-ifs`/`llvm-readtapi` from `llvmPackages_22`                       |
| python-build-standalone (rules_python)     | nixpkgs CPython (the default `python3`), one single version, in rules_python's hermetic-runtime layout |
| nightly Mojo toolchain wheel               | never fetched: `--config=build-mojo` (+ propagated to the exec config) compiles the compiler from this tree |

The glue is `nix/bazelrc.nix`, an extra `--bazelrc` carrying the
`--override_repository`/`--override_module` lines, the Nix dynamic-linker and
store rpaths for everything the toolchain links, and a fixed PATH/shell for
actions.  `nix/toolchain-repos.nix` builds the replacement repositories.
Three small accommodations for the newer toolchain live there too:
glibc's linker scripts are rewritten to bare sonames (lld refuses absolute
paths inside a sysroot), crashpad gets `-include cstdint` (gcc 15 headers no
longer include it transitively), and protobuf's protoc-authenticity check is
downgraded (it needs a shell PATH the sandbox doesn't have; protoc itself is
the module's own, compiled from source).

Build pipeline (same shape as before):

```
nix/deps.nix   ──►  fixed-output derivation: `bazel build --nobuild` of the release
     │              targets with network; output = Bazel repository cache (source
     │              archives, module registry files, pure-python helper wheels of
     │              rules_pycross — no native binaries) + MODULE.bazel.lock.
     │              The LLVM tarball is fetched by Nix itself (nix/seeds.nix).
     │
nix/sdk.nix    ──►  the real build, offline (--repository_cache,
     │              --lockfile_mode=error, --repository_disable_download):
     │              11k actions — LLVM/MLIR/Clang/LLDB + the Mojo compiler —
     │              then `mojo` compiles the standard library, all natively in
     │              the Nix sandbox.  Installed stripped (llvm-strip), laid out
     │              like the wheels' `modular/` root.
     │
nix/mojo.nix   nix/wheels.nix   nix/release.nix
```

### Purity / reproducibility properties

* No prebuilt binary is downloaded or executed anywhere in the build.  The
  repository cache of `modular-bazel-deps` contains source archives, BCR
  registry metadata and a handful of pure-Python wheels (rules_pycross's
  internal `pip`/`installer`/`packaging` helpers used at fetch time).
* Only fixed-output derivations touch the network (`modular-bazel-deps`, the
  LLVM source tarball, nixpkgs' own sources), all content-hash checked;
  everything rebuilds with `nix build --offline` afterwards.
* Bazel actions embed no timestamps (`--nostamp`, `__DATE__` redacted) and no
  absolute source paths (`-ffile-compilation-dir=.`); repeated builds of the
  SDK tree are expected to be bit-identical (verified for the previous
  toolchain generation; re-verify with `nix build --rebuild`).
* The produced binaries reference the Nix store (glibc 2.42, libstdc++ from
  gcc 15, ncurses…) via interpreter/rpath — they are real Nix binaries, which
  is why no patching step exists anywhere.

### Trade-off versus upstream binaries

Upstream compiles with clang 22.1.4 against the Ubuntu 22.04 sysroot
(glibc 2.35), producing manylinux_2_34 wheels.  This flake compiles with
nixpkgs clang 22.1.8 against nixpkgs glibc 2.42 — same compiler generation,
same code, but the artefacts are Nix-native and not portable to arbitrary
distros.  If you need manylinux artefacts bit-comparable to upstream, that
inherently requires their prebuilt clang/sysroot binaries and is out of scope
for this pure variant.

### Updating

* New source revision: nothing to do unless the external dependency set
  changed — then `nix build .#modular-bazel-deps` prints the new hash for
  `depsHashes` in `nix/default.nix`.  If `bazel/public-patches/llvm_source.bzl`
  bumps LLVM, update `nix/seeds.nix`.
* nixpkgs bump: `nix flake update` (toolchain versions move with nixpkgs).
* New release version: `python3 nix/update-release-wheels.py` refreshes
  `release-wheels.json` (only used by `release-compare`).

## Layout of the results

```
result/                          (.#mojo-sdk-unwrapped, wheel `modular/` root layout)
├── bin/  mojo lld modular-crashpad-handler mojo-lldb lldb-server lldb-dap lldb-argdumper
│         llvm-symbolizer mojo-lsp-server
├── lib/  libKGENCompilerRTShared.so libAsyncRTRuntimeGlobals.so libMSupportGlobals.so
│         libMojoLLDB.so libMojoJupyter.so liblldb24.0.0git.so mojo-repl-entry-point
│         lldb-visualizers/  mojo/std.mojoc
└── python/mojo/*.py

result/                          (.#release)
├── wheels/*.whl                 built from source
├── simple/<project>/index.html  PEP 503 index:  pip install --index-url file://$PWD/result/simple mojo
├── source/modular-<v>.tar.gz    reproducible source archive
├── SHA256SUMS
└── MANIFEST.txt
```

## Requirements

* Linux x86_64 (aarch64-linux wired, untested); no special sandbox features
  (Bazel uses its process-wrapper sandbox inside the Nix sandbox).
* `big-parallel`: ~11k C++ actions; ≥ 16 GB RAM recommended (`--jobs` is
  bounded by RAM at ~1.5 GB/job), ~20 GB scratch, roughly an hour on 16 cores.
* Editing `nix/`, flake files or top-level Markdown does not invalidate the
  compiler build (filtered out of the Bazel source tree); other changes and
  new commits (the git revision lands in `mojo --version`) do.
