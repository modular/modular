# Code examples and tests for the debugging reference

This directory contains code examples and tests for the
[Debugging reference](../../../tools/debugging-reference.mdx).

Contents:

- `tests.mojo` tests the page's guarded `breakpoint()` example.
- The `BUILD.bazel` file defines a `mojo_binary` target and a
  `modular_run_binary_test` target (`tests_test`) for `tests.mojo`.
- `test_debug_reference.sh` tests the page's tool behavior: the `mojo debug`
  command, debug levels, stack traces, `MODULAR_DEBUG`, and LLDB. Bazel
  doesn't run it.

## Run the manual tests

Most of the page describes what `mojo debug`, LLDB, and the Mojo runtime do,
which a bazel test binary can't observe. `test_debug_reference.sh` checks
each of those claims and prints `PASS` or `FAIL` with a claim number.

1. Put `mojo` on your `PATH`, for example with `pixi shell` in a Mojo
   project.

2. Run the script from any directory:

   ```bash
   ./test_debug_reference.sh
   ```

The script extracts `histogram.mojo` from the
[Debugging](../../../tools/debugging.mdx) page, as the reference page's
examples use it. It creates its files in `/tmp` and removes them on exit.

Claims that the debugging page's suite already covers
(`code/tools/debugging/test_debugging.sh`) aren't repeated here. The script's
header lists them.

## Claims the script doesn't test

- **VS Code.** Start `mojo debug --vscode` on a program that calls
  `breakpoint()`, then confirm that the **Debug Console** runs `:v` and `:bt`
  and evaluates a plain value.
- **Linux platform notes.** Each needs a broken environment: a Linux image
  without `libncurses6`, or a Docker container without `SYS_PTRACE`.

## Last run

2026-11-20, Mojo 1.3.0.dev2026111805 (310936c5):

- macOS arm64: 22 passed.
- Linux aarch64 (Ubuntu 24.04 container): 22 passed.
