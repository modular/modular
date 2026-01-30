# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --warn-on-unstable-apis emits warnings for unstable method access
# from opted-in packages.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock import StructWithMethods


fn test_stable_method():
    # Calling a stable method should not trigger a warning.
    var s = StructWithMethods()
    _ = s.stable_method()


fn test_unstable_method():
    # Calling an unstable method should trigger a warning.
    var s = StructWithMethods()
    # CHECK: warning: use of unstable API 'unstable_method'
    _ = s.unstable_method()
