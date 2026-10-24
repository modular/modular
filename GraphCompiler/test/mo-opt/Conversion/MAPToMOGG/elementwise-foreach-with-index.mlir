// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s

// CHECK-LABEL: mo.graph @add_with_index
// CHECK: mogg.routine (%{{.*}} : !mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
// CHECK-NEXT: inputs ([[IP:%[^ :,]+]])
// CHECK-NEXT: outputs ([[OP:%[^ :,]+]])
// CHECK-NEXT: context ([[CTX:%[^ :,]+]]) attributes {device = #M.device_ref<"cpu", 0>, fusionInfo = {{.*}}} {
// CHECK-NEXT:   [[ISH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8]> : !mosh.ape>
// CHECK-NEXT:   [[IT:%[^ ]+]] = mogg.tensor.create([[IP]], [[ISH]]) <device = "cpu", elementType = si32, io = Input, mut = false, shapeAttr = #mosh<ape[8]> : !mosh.ape> : !mogg.tensor
// CHECK-NEXT:   [[OSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8]> : !mosh.ape>
// CHECK-NEXT:   [[OT:%[^ ]+]] = mogg.tensor.create([[OP]], [[OSH]]) <device = "cpu", elementType = si32, io = Output, mut = true, shapeAttr = #mosh<ape[8]> : !mosh.ape> : !mogg.tensor
// CHECK-NEXT:   [[LSH:%[^ ]+]] = mogg.shape.create() <shape = #mosh<ape[8]> : !mosh.ape>
// CHECK-NEXT:   mogg.call.foreach @builtin_primitives.primitives._foreach shape([[LSH]]) context([[CTX]]) ([[IDX:%[^ :,]+]] : !map.index<1>) capturing {[[IN:%[^ :,]+]] = [[IT]] : !mogg.tensor, [[OUT:%[^ :,]+]] = [[OT]] : !mogg.tensor} device = "cpu" {
// CHECK-NEXT:     [[V:%[^ ]+]] = mogg.tensor.load [[IN]][[[IDX]]] : !mogg.tensor, !map.index<1> -> !mogg.simd<si32>
// CHECK-NEXT:     [[SUM:%[^ ]+]] = mogg.call.elementwise @testing_kernels.testing_kernels.ElemwiseAddWithIndex([[V]], [[IDX]]) moOpName = "elemwise_add_with_index" : !mogg.simd<si32>, !map.index<1> -> !mogg.simd<si32>
// CHECK-NEXT:     mogg.tensor.store [[OUT]][[[IDX]]] = [[SUM]] : !mogg.tensor, !map.index<1>, !mogg.simd<si32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
mo.graph @add_with_index(%0: !mo.tensor<[8], si32>) -> !mo.tensor<[8], si32> {
  %1 = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%0) : (!mo.tensor<[8], si32>) -> !mo.tensor<[8], si32>
  mo.output %1 : !mo.tensor<[8], si32>
}
