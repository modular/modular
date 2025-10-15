# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test for MOTO-1262: @parameter decorator formatting fix
# Validates that @parameter decorators on expression statements format correctly
# after the grammar fix.

# `mojo format` only works on `.mojo` files, and modifies them in place.
# The `grep` is used to remove the `CHECK` lines from the output so FileCheck
# doesn't match on its own directives.
# RUN: cp %s %t.mojo
# RUN: mojo format %t.mojo
# RUN: cat %t.mojo | grep -v "# CHECK:" | FileCheck %t.mojo

# CHECK: fn test_func[dtype: DType](x: Int):
# CHECK:     pass
fn test_func[dtype: DType](x: Int):
    pass

# CHECK: fn multi_param_func[T: DType, N: Int](x: Int, y: Int):
# CHECK:     pass
fn multi_param_func[T: DType, N: Int](x: Int, y: Int):
    pass

# CHECK: fn main():
# CHECK:     @parameter
# CHECK:     test_func[DType.float32](42)
# CHECK:     @parameter
# CHECK:     multi_param_func[DType.int32, 10](1, 2)
# CHECK:     fn inner():
# CHECK:         @parameter
# CHECK:         test_func[DType.float64](100)
fn main():
    # Minimal reproduction case
    @parameter
    test_func[DType.float32](42)

    # Multi-parameter case
    @parameter
    multi_param_func[DType.int32, 10](1, 2)

    # Nested context
    fn inner():
        @parameter
        test_func[DType.float64](100)
