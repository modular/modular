// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers | FileCheck %s

// A dtype-changing cast fused into a matmul epilogue (mixed precision): the
// kernel accumulates in f32, the fused cast converts to si32 on the way out.
// `AllocateAccumulatingBuffers` adds a pre-cast (f32) scratch buffer as an extra
// kernel result and points the epilogue's `$intermediateTensor` at it, so the
// accumulation goes through f32 storage while the real output stays si32.

// CHECK-LABEL: mo.graph @matmul_cast
// The kernel gains a second (scratch, f32) output alongside the si32 output.
// CHECK: map.iter.opaque
// CHECK: outputs (%{{.*}} : !mo.tensor<[2, 2], si32>, %{{.*}} : !mo.tensor<[2, 2], f32>)
// The epilogue captures the si32 output but stores through the f32 scratch
// (its `intermediate` operand), and the store itself still writes the si32 value.
// CHECK: map.iter.epilogue
// CHECK-SAME: intermediate %{{.*}} : !mo.tensor<[2, 2], f32>
mo.graph @matmul_cast(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>
) -> !mo.tensor<[2, 2], si32> {
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.cast(%mm) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  mo.output %c : !mo.tensor<[2, 2], si32>
}

// Two consumers of the matmul fuse horizontally and then into its epilogue, so
// the epilogue stores twice at two dtypes. The slice the kernel binds is built
// over `$intermediateTensor`, so it has to name the destination whose dtype the
// kernel actually hands in (f32) -- naming the si32 one points the kernel's
// accumulation at a narrower buffer and it writes past the end. A usable
// destination exists here, so no scratch result is added.

// CHECK-LABEL: mo.graph @matmul_add_and_cast
// CHECK: map.iter.opaque
// CHECK: outputs (%[[O0:.*]] : !mo.tensor<[2, 2], f32>, %[[O1:.*]] : !mo.tensor<[2, 2], si32>)
// CHECK-NOT: !mo.tensor<[2, 2], f32>, %{{.*}} : !mo.tensor<[2, 2], f32>)
// CHECK: map.iter.epilogue
// CHECK-SAME: intermediate %[[O0]] : !mo.tensor<[2, 2], f32>
// CHECK: map.store %{{.*}} : !mo.tensor<[2, 2], f32>
// CHECK: map.store %{{.*}} : !mo.tensor<[2, 2], si32>
mo.graph @matmul_add_and_cast(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>,
  %y: !mo.tensor<[2, 2], f32>
) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], si32>) {
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %s = mo.add(%mm, %y) : !mo.tensor<[2, 2], f32>
  %c = mo.cast(%mm) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  mo.output %s, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], si32>
}

// The same pair with the cast first, so the epilogue stores si32 before f32.
// The target is found by dtype, not store order: still the f32 destination,
// and still no scratch result.

// CHECK-LABEL: mo.graph @matmul_cast_and_add
// CHECK: map.iter.opaque
// CHECK: outputs (%[[O0:.*]] : !mo.tensor<[2, 2], si32>, %[[O1:.*]] : !mo.tensor<[2, 2], f32>)
// CHECK-NOT: !mo.tensor<[2, 2], f32>, %{{.*}} : !mo.tensor<[2, 2], f32>)
// CHECK: map.iter.epilogue
// CHECK-SAME: intermediate %[[O1]] : !mo.tensor<[2, 2], f32>
// CHECK: map.store %{{.*}} : !mo.tensor<[2, 2], si32>
// CHECK: map.store %{{.*}} : !mo.tensor<[2, 2], f32>
mo.graph @matmul_cast_and_add(
  %a: !mo.tensor<[2, 2], f32>,
  %b: !mo.tensor<[2, 2], f32>,
  %y: !mo.tensor<[2, 2], f32>
) -> (!mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], f32>) {
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.cast(%mm) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  %s = mo.add(%mm, %y) : !mo.tensor<[2, 2], f32>
  mo.output %c, %s : !mo.tensor<[2, 2], si32>, !mo.tensor<[2, 2], f32>
}
