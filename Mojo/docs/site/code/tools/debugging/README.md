# Code examples and tests for the debugging page

This directory contains code examples and tests for the
[Debugging](../../../tools/debugging.mdx) page.

Contents:

- `tests.mojo` tests the page's example programs (`histogram.mojo` and
  `crash.mojo`).
- The `BUILD.bazel` file defines a `mojo_binary` target and a
  `modular_run_binary_test` target (`tests_test`) for `tests.mojo`.
- `test_debugging.sh` tests the page's tool behavior: crash traces,
  breakpoints, and LLDB sessions. Bazel doesn't run it.

## Run the manual tests

Most of the page describes what `mojo debug`, LLDB, and the Mojo runtime do,
which a bazel test binary can't observe. `test_debugging.sh` checks each of
those claims and prints `PASS`, `FAIL`, or `SKIP` with a claim number.

1. Put `mojo` on your `PATH`, for example with `pixi shell` in a Mojo
   project.

2. Run the script from any directory:

   ```bash
   ./test_debugging.sh
   ```

The script extracts `histogram.mojo`, `crash.mojo`, and the stderr
sample from the page, so the code and line numbers it tests match the
page. It creates its files in `/tmp` and removes them on exit.

Run it on macOS and on Linux. Some claims differ by platform, and the script
checks the expected result for the platform it runs on. In a Docker
container, test 30 (attach by PID) reports `SKIP` unless the container runs
with `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined`.

## Claims the script doesn't test

- **Debug in VS Code.** Start `mojo debug --vscode` on a program that calls
  `breakpoint()`, then confirm that VS Code pauses, that **Continue** resumes
  the program, and that the **Debug Console** runs `:v` and `:bt`.
- **Debug GPU code.** The section links to another page.
- **Troubleshooting on Linux.** Each entry needs a broken environment: a
  Linux image without `libncurses6`, or a Docker container without
  `SYS_PTRACE`. Tests 34 and 35 cover the compilation cache entry.

## Last run

2026-11-23, Mojo 1.3.0.dev2026111805 (310936c5):

- macOS arm64: 39 passed, 1 skipped (attach denied by the sandbox).
- Linux aarch64 (Ubuntu 24.04 container): 39 passed, 1 skipped (no
  `SYS_PTRACE`). With `SYS_PTRACE`, attach passes.
