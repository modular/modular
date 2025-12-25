# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

##===----------------------------------------------------------------------===##
# __comptime_assert
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"test_assert_with_message
fn test_assert_with_message[cond: Bool]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>, data_to_str({{.*}}"custom error message"
    __comptime_assert cond, "custom error message"


# CHECK-LABEL: lit.fn @"test_assert_with_long_message
fn test_assert_with_long_message[cond: Bool]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>, data_to_str({{.*}}"custom error message with long message and more"
    __comptime_assert cond, "custom error message"
                            " with long message "
                            "and more"


# CHECK-LABEL: lit.fn @"test_assert_with_message_parameter
fn test_assert_with_message_parameter[x: Int]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Int x, "_mlir_value">{{.*}}>, data_to_str{{.*}}@String::@"__init__
    __comptime_assert x, String(x)


# CHECK-LABEL: lit.fn @"test_assert_with_param_expr
fn test_assert_with_param_expr[x: Int, y: Int]():
    # CHECK: kgen.param.assert <{{.*}}eq(#lit.struct.extract<:!Int x, "_mlir_value">, #lit.struct.extract<:!Int y, "_mlir_value">){{.*}}>
    __comptime_assert x == y


fn requires_natural[x: Int](y: Int) where x >= 0:
    pass


# CHECK-LABEL: lit.fn @"test_assert_enables_where_constraint
fn test_assert_enables_where_constraint[x: Int](y: Int):
    # First assert that x >= 0
    # CHECK: kgen.param.assert <{{.*}}ge(#lit.struct.extract<:!Int x, "_mlir_value">, 0){{.*}}>
    __comptime_assert x >= 0

    # Now we can call a function that requires x >= 0 via where clause
    # CHECK: lit.call {{.*}}@"requires_natural
    requires_natural[x](y)


# CHECK-LABEL: lit.fn @"test_always_true_warning
fn test_always_true_warning():
    # CHECK-NOT: kgen.param.assert
    __comptime_assert 2 > 1, "this assert is useless"
