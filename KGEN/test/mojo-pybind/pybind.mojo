# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -gen-pybind --mojo-enable-prebuilt-packages | FileCheck %s

# CHECK-LABEL: lit.file_module @pybind

# CHECK-LABEL: lit.fn @"PyInit_impl_pybind
# CHECK-SAME: (?, %error: !lit.ref<!Error, {{.*}}, %result: !lit.ref<!PythonObject, {{.*}}) throws -> i1
# CHECK-NEXT: [[MODULE:%.*]] = lit.var.decl {{.*}}TypedPythonObject{{.*}}Module
# CHECK-NEXT: lit.call {{.*}}create_pybind_module{{.*}}"pybind"{{.*}}(%error, [[MODULE]])
# CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}PythonObject::@"__init__{{.*}}([[MODULE]])
# CHECK-NEXT: lit.ref.store [[TMP]], %result
# CHECK-NEXT: [[NONE_DEST:%.*]] = lit.var.decl
# CHECK-NEXT: lit.call {{.*}}gen_pytype_wrapper{{.*}}<:!Pythonable #Int{{[0-9]}}, :!StringLiteral {:string "Int"}>(%result, %error, [[NONE_DEST]])
# CHECK-NEXT: [[NONE_DEST2:%.*]] = lit.var.decl "anonymous*" synth : !lit.ref<none, mut *"anonymous*`2">
# CHECK-NEXT: lit.call {{.*}}add_wrapper_to_module{{.*}}@pybind::@"arg_reg_trivial_borrowed__wrapper({{.*}}"arg_reg_trivial_borrowed"{{.*}}>(%result, %error, [[NONE_DEST2]])
# CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
# CHECK-NEXT: return [[FALSE]]

# CHECK-LABEL: lit.fn export C @"PyInit_pybind
# CHECK-SAME: () -> !PythonObject
# CHECK-SAME: linkageName = "PyInit_pybind"
# CHECK-NEXT: %module = lit.var.decl {{.*}}!PythonObject
# CHECK-NEXT: %error = lit.var.decl {{.*}}!Error
# CHECK-NEXT: lit.try %error
# CHECK-NEXT:   call @pybind::@"PyInit_impl_pybind{{.*}}(%error, %module)
# CHECK-NEXT:   lit.try.yield
# CHECK-NEXT: except
# CHECK-NEXT:   [[NULL:%.*]] = lit.call {{.*}}fail_initialization{{.*}}(%error)
# CHECK-NEXT:   return [[NULL]]
# CHECK:      [[SR:%.*]] = lit.load.consume %module
# CHECK-NEXT: return [[SR]]


fn arg_reg_trivial_borrowed(arg: Int):
    pass
