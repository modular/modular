# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Sub-package of test_std_mock for testing intra-package stability checks.
# Code here uses unstable APIs from the parent package (test_std_mock).
# Since both are sub-packages of the same opted-in package, no warnings
# should be emitted.

from test_std_mock import UnstableStruct


@stable
fn subpkg_stable_fn() -> Int:
    # This uses UnstableStruct from the parent package.  Both this sub-package
    # and the parent are under the opted-in "test_std_mock" package, so this
    # is intra-package usage and should NOT warn.
    var s = UnstableStruct()
    return s.unstable_method()
