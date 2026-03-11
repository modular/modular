# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s --verify-diagnostics

# ===----------------------------------------------------------------------=== #
# __is_run_in_comptime_interpreter
# ===----------------------------------------------------------------------=== #


fn test_with_comptime_if():
    #expected-error@+1 {{cannot use a dynamic value in 'comptime if' condition}}
    comptime if __is_run_in_comptime_interpreter:
        var x: Int

fn test_as_comptime_expression[b: Bool]():
    #expected-error@+1 {{cannot use a dynamic value in comptime initializer}}
    comptime i = __is_run_in_comptime_interpreter and b:
