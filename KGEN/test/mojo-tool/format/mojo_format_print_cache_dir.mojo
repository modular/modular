# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo format --print-cache-dir 2>&1 | FileCheck %s
# Assumes MBLACK_CACHE_DIR is unset, so the default path contains "mblack".
# CHECK: mblack
