# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This test validates the fix for MOTO-1262 with complex nested @parameter decorators
# that previously failed to parse. The formatter should now handle these cases correctly.

# `mojo format` only works on `.mojo` files, and modifies them in place.
# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cp %s %t.mojo
# RUN: mojo format %t.mojo
# RUN: cat %t.mojo | grep -v "# CHECK:" | FileCheck %t.mojo

# CHECK: fn some_wrapper_function[
# CHECK:     param1: Int,
# CHECK:     param2: Int,
# CHECK: ](arg1: Int, arg2: Int, arg3: Int,):
# CHECK:     pass
fn some_wrapper_function[
    param1: Int,
    param2: Int,
](
    arg1: Int,
    arg2: Int,
    arg3: Int,
):
    pass

# CHECK: fn test_parameter_parsing_issue():
# CHECK:     @parameter
# CHECK:     @always_inline
# CHECK:     fn inner_function[simd_width: Int, rank: Int](idx: Int):
# CHECK:         @parameter
# CHECK:         some_wrapper_function[param1=123, param2=456,](
# CHECK:             arg1=1,
# CHECK:             arg2=2,
# CHECK:             arg3=3,
# CHECK:         )
# CHECK:     inner_function[1, 2](0)
fn test_parameter_parsing_issue():
    @parameter
    @always_inline
    fn inner_function[
        simd_width: Int, rank: Int
    ](idx: Int):
        @parameter
        some_wrapper_function[
            param1=123,
            param2=456,
        ](
            arg1=1,
            arg2=2,
            arg3=3,
        )

    inner_function[1, 2](0)
