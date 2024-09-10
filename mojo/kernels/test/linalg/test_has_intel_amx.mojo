# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: linux, intel_amx
# RUN: %mojo-no-debug %s

from sys import has_intel_amx, os_is_linux

from linalg.intel_amx_intrinsics import init_intel_amx
from testing import assert_false, assert_true


fn test_has_intel_amx() raises:
    assert_true(os_is_linux())
    assert_true(has_intel_amx())
    assert_true(init_intel_amx())


fn main() raises:
    test_has_intel_amx()
