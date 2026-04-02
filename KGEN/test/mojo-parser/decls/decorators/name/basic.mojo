# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @__name sets the linkage name without exporting.

# CHECK: lit.fn @"my_func()"
# CHECK-SAME: linkageName = "custom_name" : !kgen.string
# CHECK-NOT: export
@__name("custom_name")
def my_func():
    pass


# Test that @__name with a parametric expression produces a param.expr.

comptime a = 42

# CHECK: lit.fn @"parametric_name[::Int]()"
# CHECK-SAME: linkageName = #kgen.param.expr<data_to_str
@__name("prefix_" + String(T) + "_" + String(a))
def parametric_name[T: Int]():
    pass


# Test that @__name combined with @export sets the linkage name and exports.
# Order doesn't matter: @__name sets the linkage name, @export marks exported.

# CHECK: lit.fn export @"name_then_export()"
# CHECK-SAME: linkageName = "my_export" : !kgen.string
@__name("my_export")
@export
def name_then_export():
    pass


# CHECK: lit.fn export C @"name_then_c_export()"
# CHECK-SAME: linkageName = "my_c_export" : !kgen.string
@export(ABI="C")
@__name("my_c_export")
def name_then_c_export():
    ...


# Same but with reversed order.


# CHECK: lit.fn export @"export_then_name()"
# CHECK-SAME: linkageName = "my_export2" : !kgen.string
@export
@__name("my_export2")
def export_then_name():
    pass


# CHECK: lit.fn export C @"c_export_then_name()"
# CHECK-SAME: linkageName = "my_c_export2" : !kgen.string
@export(ABI="C")
@__name("my_c_export2")
def c_export_then_name():
    ...
