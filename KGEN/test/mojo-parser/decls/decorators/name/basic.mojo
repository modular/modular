# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @__name sets the linkage name without exporting.


# CHECK: lit.fn @"my_func()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"custom_name" : !kgen.string, false>
# CHECK-NOT: export
@__name("custom_name")
def my_func():
    pass


# Test that @__name with a parametric expression produces a param.expr.

comptime a = 42


# CHECK: lit.fn @"parametric_name[::Int]()"
# CHECK-SAME: linkageName = #kgen.linkage_name<#kgen.param.expr<data_to_str
@__name("prefix_" + String(T) + "_" + String(a))
def parametric_name[T: Int]():
    pass


# Test that mangle=True is stored in the IR as the boolean true.


# CHECK: lit.fn @"mangle_true()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_mangled" : !kgen.string, true>
@__name("my_mangled", mangle=True)
def mangle_true():
    pass


# Test that explicit mangle=False is identical to omitting the argument.


# CHECK: lit.fn @"mangle_false()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_unmangled" : !kgen.string, false>
@__name("my_unmangled", mangle=False)
def mangle_false():
    pass


# Test that @__name combined with @export sets the linkage name and exports.
# Order doesn't matter: @__name sets the linkage name, @export marks exported.


# CHECK: lit.fn export @"name_then_export()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_export" : !kgen.string, false>
@__name("my_export")
@export
def name_then_export():
    pass


# CHECK: lit.fn export C @"name_then_c_export()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_c_export" : !kgen.string, false>
@export(ABI="C")
@__name("my_c_export")
def name_then_c_export():
    ...


# Same but with reversed order.


# CHECK: lit.fn export @"export_then_name()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_export2" : !kgen.string, false>
@export
@__name("my_export2")
def export_then_name():
    pass


# CHECK: lit.fn export C @"c_export_then_name()"
# CHECK-SAME: linkageName = #kgen.linkage_name<"my_c_export2" : !kgen.string, false>
@export(ABI="C")
@__name("my_c_export2")
def c_export_then_name():
    ...
