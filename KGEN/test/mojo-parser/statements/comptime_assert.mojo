# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

##===----------------------------------------------------------------------===##
# comptime assert
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"test_assert_with_message
def test_assert_with_message[cond: Bool]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>, data_to_str({{.*}}"custom error message"
    comptime assert cond, "custom error message"


# CHECK-LABEL: lit.fn @"test_assert_with_long_message
def test_assert_with_long_message[cond: Bool]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>, data_to_str({{.*}}"custom error message with long message and more"
    comptime assert cond, "custom error message"
                          " with long message "
                          "and more"


# CHECK-LABEL: lit.fn @"test_assert_with_message_parameter
def test_assert_with_message_parameter[x: Int]():
    # CHECK: kgen.param.assert <{{.*}}#lit.struct.extract<:!Int x, "_mlir_value">{{.*}}>, data_to_str{{.*}}@String::@"__init__
    comptime assert x, String(x)


# CHECK-LABEL: lit.fn @"test_assert_with_param_expr
def test_assert_with_param_expr[x: Int, y: Int]():
    # CHECK: kgen.param.assert <{{.*}}eq(#lit.struct.extract<:!Int x, "_mlir_value">, #lit.struct.extract<:!Int y, "_mlir_value">){{.*}}>
    comptime assert x == y


def requires_natural[x: Int](y: Int) where x >= 0:
    pass


# CHECK-LABEL: lit.fn @"test_assert_enables_where_constraint
def test_assert_enables_where_constraint[x: Int](y: Int):
    # First assert that x >= 0
    # CHECK: kgen.param.assert <{{.*}}ge(#lit.struct.extract<:!Int x, "_mlir_value">, 0){{.*}}>
    comptime assert x >= 0

    # Now we can call a function that requires x >= 0 via where clause
    # CHECK: lit.call {{.*}}@"requires_natural
    requires_natural[x](y)


# CHECK-LABEL: lit.fn @"test_assert_with_tstring_message
def test_assert_with_tstring_message[x: Int]():
    # CHECK: kgen.param.assert <{{.*}}>, data_to_str({{.*}}__make_tstring
    comptime assert x, t"expected positive, got {x}"


# CHECK-LABEL: lit.fn @"test_always_true_warning
def test_always_true_warning():
    # CHECK-NOT: kgen.param.assert
    comptime assert 2 > 1, "this assert is useless"
