# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --warn-on-unstable-apis emits warnings when an unstable function
# is referenced (not called). This is a regression test for a gap where function
# references bypassed the stability check.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock import stable_fn, unstable_fn


def takes_fn_ref(f: def() thin -> None):
    """Helper function that takes a function reference."""
    f()


def test_stable_fn_reference():
    """Referencing a stable function should not trigger a warning."""
    # No warning expected here.
    var f = stable_fn
    takes_fn_ref(f)


def test_unstable_fn_reference():
    """Referencing an unstable function (not calling it) should trigger a warning."""
    # CHECK: warning: use of unstable API 'unstable_fn'
    var f = unstable_fn
    takes_fn_ref(f)
