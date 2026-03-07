# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests that assert desugars to a call to debug_assert at the parser level,
# and that the lowered IR contains a conditional check with a trap.
#
# Verify parser output: assert produces a call to debug_assert (no lit.assert op).
# RUN: %parse-mojo-isolated -D ASSERT=all %s | FileCheck %s --check-prefix=PARSE
#
# Verify lowered output: the debug_assert call lowers to a conditional with a trap.
# RUN: %parse-mojo-isolated -D ASSERT=all %s | kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit | FileCheck %s --check-prefix=LOWER


# PARSE-LABEL: lit.fn @"test_assert_basic
fn test_assert_basic(cond: Bool):
    # PARSE: debug_assert
    assert cond


# PARSE-LABEL: lit.fn @"test_assert_with_message
fn test_assert_with_message(cond: Bool):
    # PARSE: debug_assert
    assert cond, "something went wrong"


# After lowering, the debug_assert call should produce conditional checks.
# LOWER: hlcf.if
# LOWER: kgen.unreachable
