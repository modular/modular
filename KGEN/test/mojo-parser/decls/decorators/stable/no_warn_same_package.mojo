# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --warn-on-unstable-apis does NOT warn for same-package usage.
#
# Phase 4.3: Usage within the same opted-in package should not trigger warnings.
# This allows library authors to freely use unstable internal APIs.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock import stable_fn_using_unstable


def test_stable_wrapper():
    # Calling a stable function that internally uses unstable APIs should NOT
    # warn. The user is calling the stable API; they don't need to know about
    # or be warned about the internal implementation details.
    #
    # CHECK-NOT: warning: use of unstable API
    var result = stable_fn_using_unstable()
    _ = result
