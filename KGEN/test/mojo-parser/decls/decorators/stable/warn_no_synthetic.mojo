# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that compiler-generated members don't trigger stability warnings

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

# CHECK-NOT: warning: stable struct 'StableStructWithMovable' implements stable trait method '__init__' with unstable implementation
from test_std_mock import StableStructWithMovable

def main():
    pass