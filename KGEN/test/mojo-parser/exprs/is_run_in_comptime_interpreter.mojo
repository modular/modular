# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test __is_run_in_comptime_interpreter bare keyword.
# Verifies that it emits kgen.is_run_in_comptime_interpreter : i1 directly.
# It is a runtime value and is intended for use in runtime 'if', not
# 'comptime if' (which requires a compile-time-parametric condition).


# CHECK-LABEL: lit.fn @"test_basic
fn test_basic():
    # CHECK: kgen.is_run_in_comptime_interpreter : i1
    var x = __is_run_in_comptime_interpreter
    _ = x


# CHECK-LABEL: lit.fn @"test_in_runtime_if
fn test_in_runtime_if():
    # CHECK: kgen.is_run_in_comptime_interpreter : i1
    if __is_run_in_comptime_interpreter:
        pass
