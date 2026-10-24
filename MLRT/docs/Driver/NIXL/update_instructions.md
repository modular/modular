# NIXL Update Instructions

We vendor [upstream NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA's
Inference Xfer Library) as an unmodified Bazel `http_archive`
(`@nixl_upstream`) pinned to a released tag, the same way we vendor LLVM. This
document covers how to bump the pinned version and how to carry a local patch
if one is ever needed.

## How to bump the upstream NIXL version

The version is pinned in a `BEGIN_NIXL_GENERATED` / `END_NIXL_GENERATED`
block inside `oss/modular/bazel/common.MODULE.bazel` (the `nixl_upstream`
`http_archive`: its tag, URL, `strip_prefix`, and `sha256`). Use the
`update-nixl` helper to update it:

```bash
git checkout -b nixl-bump main
git pull --rebase origin main
update-nixl 1.2.0           # rewrite the pin and print the new sha256
update-nixl 1.2.0 --commit  # same, plus create a git commit
```

`update-nixl` downloads the release tarball for the given tag from
`github.com/ai-dynamo/nixl`, computes its `sha256`, and rewrites the generated
block in `oss/modular/bazel/common.MODULE.bazel`. The script lives at
`utils/update_nixl.py` and mirrors `utils/update_llvm.py`.

If upstream's source layout or build inputs changed, you may also need to
update the Bazel overlay at `bazel/public-patches/nixl_upstream.BUILD`. It
lists every `.cpp` explicitly and globs only headers. The plugin `.so`s link
with `-Wl,-z,undefs`, so a source file that upstream added and the overlay
missed does not fail the link. It fails at `dlopen` time, or stays quiet
until something calls the missing symbol. Diff each library's source list
against the corresponding `meson.build` on every bump. After bumping,
rebuild and re-test (see [Testing](#testing-after-an-update)).

## Bumping the Rust bindings in lockstep

The Rust side consumes NIXL through the `nixl-sys` crate, pinned in
`dkv/Cargo.toml` and locked in both `dkv/Cargo.lock` and `mach/Cargo.lock`.
Move it to the same version as `@nixl_upstream`, or the crate's headers and
`libnixl.so` disagree at the symbol level.

Bazel disables the crate's build script, so the bindgen output it would
normally produce is vendored by
`bazel/third-party/nixl-sys-vendor-bindings.patch`. Regenerate that patch
whenever the crate's `wrapper.h` changes, which a version bump usually
implies. Generate the bindings with the same tool the existing patch used,
then rebuild the patch as a `diff -ruN` between a pristine copy of the crate
and a copy carrying the generated file:

```bash
cargo install bindgen-cli --version 0.71.1 --locked
(cd dkv && cargo fetch)  # unpacks the crate into the registry

repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
crates=(~/.cargo/registry/src/*/nixl-sys-<version>)
[[ ${#crates[@]} -eq 1 ]] || { echo "expected 1 dir, got ${crates[*]}"; exit 1; }
cp -r "${crates[0]}" "$work/a" && cp -r "${crates[0]}" "$work/b"
(cd "$work/b" && bindgen wrapper.h -o src/bindings.rs -- -std=c++20 -x c++)
sed -i 's|include!(concat!(env!("OUT_DIR"), "/bindings.rs"));|include!("bindings.rs");|' \
  "$work/b/src/lib.rs"
(cd "$work" && diff -ruN a/src b/src) \
  > "$repo/bazel/third-party/nixl-sys-vendor-bindings.patch"
```

Four things about that recipe matter:

- The `lib.rs` hunk is what lets the crate find the vendored file with its
  build script disabled. Keep it.
- Run `diff` from inside the scratch directory, with the two copies named
  `a` and `b`, so the paths in the patch are the `a/src/...` and `b/src/...`
  that `patch -p1` expects.
- `bindgen` formats its output with `rustfmt`, resolved through the
  `rust-toolchain.toml` in the crate copy. Run it from inside `$work/b`, not
  from the monorepo, or you get the monorepo's toolchain instead.
- The output depends on the libclang that `bindgen` loads. A stray
  `pub type rsize_t = ...` typedef from the host's libc headers is the tell;
  the committed patch has no such line. If yours does, drop it.

## Dealing with NIXL patches

We depend on **unmodified** upstream NIXL. If you must carry a local change
(for example, a fix that has not landed in a tagged release yet), apply it as a
patch rather than forking:

1. Drop the patch file at `oss/modular/bazel/public-patches/nixl-<name>.patch`
   (the `nixl-` prefix matches the convention for other vendored patches). This
   directory is OSS-visible and resolves in both the internal build and the
   standalone OSS-mirror build; do NOT put it in `bazel/third-party/` (that
   directory is internal-only and the label would not resolve in the OSS build).
2. Add it to the `patches = [...]` list of the `nixl_upstream` `http_archive` in
   `oss/modular/bazel/common.MODULE.bazel`. The archive already sets
   `patch_args = ["-p1"]`. Use the label form
   `"//bazel/public-patches:nixl-<name>.patch"`.
3. The `update-nixl` script (`utils/update_nixl.py`) **carries the existing
   patches list through** when rewriting the generated block on a version bump —
   it does not reset `patches = []`. After bumping, verify that each carried
   patch still applies against the new tarball. Remove the patch entry and file
   when the fix lands upstream.

> [!NOTE]
> Every patch should include a `TODO` comment referencing the upstream tracking
> issue or PR (e.g. `TODO(GEX-NNNN): drop once fix is upstreamed`) so it is
> clear when the patch can be removed. The point of patching is to unblock
> quickly, not to maintain a fork.

## Testing after an update

A NIXL change must be validated against the in-tree libraries, the
disaggregated-inference stack, and EFA. See
[`testing.md`](testing.md) for the lane overview and
[`validate_upstream_sync.md`](validate_upstream_sync.md) for the full
pin-bump procedure and test list.

The quickest path is the one-click GitHub Actions workflow, which runs the
NIXL unit tests (Core/UCX on a B200) and dispatches the shared-pool tests,
including the nightly's `nixl-validation-1p1d-*` Mammoth benchmarks:

```bash
gh workflow run validateNIXLSync.yaml \
  --repo modularml/modular \
  --ref <branch-or-main>
```

Running from any branch — including `main` — auto-builds the MAX Serve images
from that commit and passes them to the shared-pool tests — no manual image
build step needed.

As a final sanity check, also run `br //:install`.
