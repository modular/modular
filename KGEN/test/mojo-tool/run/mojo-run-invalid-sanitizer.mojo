# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Note: Don't run with pre-existing sanitizers to ensure sanitizers work in a
#       clean environment.
# UNSUPPORTED: asan,msan,tsan

# RUN: not mojo --sanitize address %s 2>&1 | FileCheck %s --check-prefix=CHECK_ADDRESS
# RUN: not mojo --sanitize thread %s 2>&1 | FileCheck %s --check-prefix=CHECK_THREAD

# CHECK_ADDRESS: This build of `mojo` does not support `mojo run` with `--sanitize address`, consider generating a sanitized executable using `mojo build` instead.
# CHECK_THREAD: does not support `mojo run` with `--sanitize thread`
