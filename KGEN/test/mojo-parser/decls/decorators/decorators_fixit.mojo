# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests for @implicit(deprecated=True) fixit suggestions.
# Uses JSON diagnostic format to verify exact fixit positions.

# RUN: %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Test: Deprecated implicit conversion fixit wraps expression with explicit call
# ===----------------------------------------------------------------------=== #


struct DeprecatedImplicit:
    @implicit(deprecated=True)
    fn __init__(out self, value: Int):
        pass


fn takes_deprecated_implicit(x: DeprecatedImplicit):
    pass


fn main():
    # Variable assignment with implicit conversion.
    # The fixit suggests wrapping the literal with the explicit constructor call.
    # The fixIts are on the note diagnostic, not the warning.
    # CHECK: "fixIts":[{"end":{"column":29,"line":[[#@LINE+2]]},"start":{"column":29,"line":[[#@LINE+2]]},"text":"DeprecatedImplicit("},{"end":{"column":30,"line":[[#@LINE+2]]},"start":{"column":30,"line":[[#@LINE+2]]},"text":")"}]
    # CHECK-SAME: "message":"call 'DeprecatedImplicit(...)' explicitly"
    _: DeprecatedImplicit = 1

    # Function argument with implicit conversion.
    # The fixit suggests wrapping Int(1) with the explicit constructor call.
    # CHECK: "fixIts":[{"end":{"column":31,"line":[[#@LINE+2]]},"start":{"column":31,"line":[[#@LINE+2]]},"text":"DeprecatedImplicit("},{"end":{"column":37,"line":[[#@LINE+2]]},"start":{"column":37,"line":[[#@LINE+2]]},"text":")"}]
    # CHECK-SAME: "message":"call 'DeprecatedImplicit(...)' explicitly"
    takes_deprecated_implicit(Int(1))
