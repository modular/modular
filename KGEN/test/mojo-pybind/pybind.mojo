# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -gen-pybind --mojo-enable-prebuilt-packages | FileCheck %s

# CHECK-LABEL: lit.file_module @pybind

# CHECK-LABEL: lit.func @"PyInit_impl_pybind
# CHECK-SAME: (?, %error: !lit.ref<!Error, {{.*}}, %result: !lit.ref<!PythonObject, {{.*}}) throws -> i1
# CHECK-NEXT: [[MODULE:%.*]] = lit.var.decl {{.*}}TypedPythonObject{{.*}}Module
# CHECK-NEXT: call {{.*}}create_pybind_module{{.*}}"pybind"{{.*}}(%error, [[MODULE]])
# CHECK-NEXT: [[SR:%.*]] = lit.load.consume [[MODULE]]
# CHECK-NEXT: call {{.*}}PythonObject::@"__init__{{.*}}(%result, [[SR]])
# CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
# CHECK-NEXT: return [[FALSE]]

# CHECK-LABEL: lit.func export C @"PyInit_pybind
# CHECK-SAME: () -> !PythonObject
# CHECK-SAME: linkageName = "PyInit_pybind"
# CHECK-NEXT: %module = lit.var.decl {{.*}}!PythonObject
# CHECK-NEXT: %error = lit.var.decl {{.*}}!Error
# CHECK-NEXT: lit.try %error
# CHECK-NEXT:   call @pybind::@"PyInit_impl_pybind{{.*}}(%error, %module)
# CHECK:      [[SR:%.*]] = lit.load.consume %module
# CHECK-NEXT: return [[SR]]
