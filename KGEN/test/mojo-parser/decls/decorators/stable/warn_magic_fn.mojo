# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --warn-on-unstable-apis emits warnings when unstable magic functions
# are used. Some magic functions (origin_of, type_of, conforms_to,
# __functions_in_module) are considered stable and should not warn.

# RUN: %parse-mojo-isolated -warn-on-unstable-apis %s 2>&1 | FileCheck %s


fn test_stable_magic_functions():
    """Stable magic functions should NOT trigger warnings."""
    var x = 42
    # type_of, origin_of, conforms_to, __functions_in_module are stable.
    # CHECK-NOT: warning:{{.*}}type_of
    comptime t = type_of(x)


fn test_unstable_get_current_function_name():
    """Using __get_current_function_name should trigger an unstable warning."""
    # CHECK: warning: use of unstable function '__get_current_function_name'
    comptime name = __get_current_function_name()
