// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mogg-dump-kernels 2>&1 | FileCheck %s

// Verify that mogg-dump-kernels uses KernelDeclAdaptor when mogg.kernel.decl
// ops are present (produced by mogg-import-packages).

// CHECK-DAG: @register(imposter_add)
// CHECK-DAG: @register(imposter_add_lhs)
// CHECK-DAG: @register(imposter_add_elementwise)
// CHECK-DAG: @primitive(mogg.call.foreach)

module {}
