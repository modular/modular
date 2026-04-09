# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that intra-package usage across sub-packages does NOT warn.
#
# Code in test_std_mock.subpkg uses unstable APIs from test_std_mock.
# Both are under the same opted-in package ("test_std_mock"), so the
# compiler must recognize them as the same package and suppress warnings.
# This requires walking the full ancestor PackageOp chain, not just
# checking the nearest (leaf) package name.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock.subpkg import subpkg_stable_fn


def test():
    # The function itself is @stable, so no warning for calling it.
    # Internally it uses unstable APIs from a sibling sub-package,
    # which should also not warn (intra-package).
    #
    # CHECK-NOT: warning: use of unstable API
    _ = subpkg_stable_fn()
