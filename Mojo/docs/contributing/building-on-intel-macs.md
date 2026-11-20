# Building on Intel Macs

> [!WARNING]
> Intel Macs (macOS on x86_64) are community-supported only. Modular does not
> ship Mojo or MAX for them and does not test them in CI. You build everything
> from source, and it can break when `main` changes.

The Bazel build normally downloads a prebuilt LLVM toolchain and tcmalloc.
Modular only provides these for Apple silicon and Linux. On an Intel Mac, you
build them once with a script in this repository. Everything else uses the
normal `./bazelw` build.

## Requirements

- An Intel Mac with an AVX2-capable CPU, which means roughly 2013 or newer
  (Haswell or later). Mojo targets `x86-64-v3` on this platform.
- The Xcode command line tools (Apple clang and a macOS SDK).
- `cmake` and `ninja` on your `PATH`.
- A network connection for the first setup, and several GB of free disk space.

## Build

1. From the repository root, build the toolchain:

   ```bash
   bazel/internal/cc-toolchain/intel-mac/bootstrap.sh
   ```

   It builds LLVM and tcmalloc into `build/intel-mac/toolchain`. Git ignores
   this directory. To use another directory, pass it as an argument or set
   `WORK_DIR`.

2. Add the line that the script prints to `local.bazelrc` in the repository
   root. Create the file if it does not exist. The line looks like this:

   ```text
   common --repo_env=MODULAR_INTEL_MAC_TOOLCHAIN=/path/to/build/intel-mac/toolchain
   ```

3. Build as usual, for example:

   ```bash
   ./bazelw build //Mojo/examples:hello
   ```

If the pinned LLVM version changes, run the script again.

If the build fails while loading Python packages, add this line to
`local.bazelrc`:

```text
build --config=disable-mypy
```

Without the setting from step 2, `bazel query` and analysis still work, but
compiling or linking fails with a message that tells you to run the script.

## Limits

- Community-supported, with no CI. Expect breakage.
- The first build is slow. On one 6-core i7 (i7-9750H), LLVM took about an
  hour and the first Bazel build took about two more.
- The lint and format tools (`ruff`, `rumdl` and `shellcheck`) are not set up
  for Intel Macs.
- Some targets need Python wheels that the pip lock file may not have for
  Intel Macs, so build the specific targets you need instead of `//...`.
- The public Bazel cache has nothing built for Intel Macs, so you will rarely
  get cache hits.

## Tips

### Follow a release branch

The pinned LLVM version changes on `main` from time to time, and each change
means rebuilding the toolchain and everything that depends on it. A
`max/v<version>` release branch keeps the pin it was cut with, so you only
rebuild when you switch to a branch with a different pin.

### Turn off the remote cache

Set `BAZEL_DISABLE_REMOTE_CACHE=1` to skip the public remote cache, which is
read-only by default:

```bash
export BAZEL_DISABLE_REMOTE_CACHE=1
```
