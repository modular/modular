// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-outline-structured-kernel-bodies --map-to-mogg | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-outline-structured-kernel-bodies --map-to-mogg | mo-opt

// A rowwise kernel lowers to a `mogg.call.rowwise` over its shape and axis,
// and each fold to a `mogg.rowwise.reduce` that yields its contribution.

// CHECK-LABEL: mo.graph @fused
// CHECK:       mogg.routine
// CHECK:       %[[X:.*]] = mogg.tensor.create({{.*}}elementType = bf16, io = Input
// CHECK:       %[[B:.*]] = mogg.tensor.create({{.*}}elementType = f32, io = Input
// CHECK:       %[[OUT:.*]] = mogg.tensor.create({{.*}}io = Output
// CHECK:       %[[SHAPE:.*]] = mogg.shape.create() <shape = #mosh<ape[8, 128]> : !mosh.ape>
// CHECK-NEXT:  mogg.call.rowwise shape(%[[SHAPE]]) axis(1) context(%{{.*}}) (%row : !map.index<2>) capturing {%cap0 = %[[X]] : !mogg.tensor, %cap1 = %[[B]] : !mogg.tensor, %cap2 = %[[OUT]] : !mogg.tensor} device = "cpu" {
// CHECK-NEXT:    %[[ACC:.*]] = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
// CHECK-NEXT:      %[[V:.*]] = mogg.tensor.load %cap0[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<bf16>
// CHECK-NEXT:      %[[A:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Cast(%[[V]])
// CHECK-NEXT:      %[[SQ:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Mul(%[[A]], %[[A]])
// CHECK-NEXT:      mogg.yield %[[SQ]] : !mogg.simd<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[BV:.*]] = mogg.tensor.load %cap1[%row] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[Y:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Add(%[[ACC]], %[[BV]])
// CHECK-NEXT:    mogg.tensor.store %cap2[%row] = %[[Y]] : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
// CHECK-NEXT:  }
mo.graph @fused(%x: !mo.tensor<[8, 128], bf16>, %b: !mo.tensor<[8, 1], f32>) -> !mo.tensor<[8, 1], f32> {
  %a = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %sq = mo.mul(%a, %a) : !mo.tensor<[8, 128], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %y = mo.add(%r, %b) : !mo.tensor<[8, 1], f32>
  mo.output %y : !mo.tensor<[8, 1], f32>
}

// Dynamic dims and a non-inner axis; one store per output field.
// CHECK-LABEL: mo.graph @argMaxDynamic
// CHECK:       dims (%[[M:.*]]: !mogg.dim<M>, %[[N:.*]]: !mogg.dim<N>)
// CHECK:       mogg.tensor.create({{.*}}elementType = si64
// CHECK:       %[[SHAPE2:.*]] = mogg.shape.create(%[[M]], %[[N]]) <shape = #mosh<ape[dyn, dyn]> : !mosh.ape>
// CHECK-SAME:  !mogg.dim<M>, !mogg.dim<N>
// CHECK-NEXT:  mogg.call.rowwise shape(%[[SHAPE2]]) axis(0)
// CHECK-NEXT:    %[[F:.*]]:2 = mogg.rowwise.reduce @builtin_kernels.monoids.ArgMax (%i : !map.index<2>) -> (!mogg.simd<f32>, !mogg.simd<si64>) {
// CHECK:         mogg.tensor.store %cap1[%row] = %[[F]]#0 : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
// CHECK-NEXT:    mogg.tensor.store %cap2[%row] = %[[F]]#1 : !mogg.tensor, !map.index<2>, !mogg.simd<si64>
mo.graph @argMaxDynamic<M, N>(%x: !mo.tensor<[M, N], f32>) -> (!mo.tensor<[1, N], f32>, !mo.tensor<[1, N], si64>) {
  %v, %i = mo.custom [symbol = "mo.reduce.arg_max.tmp", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[M, N], f32>) -> (!mo.tensor<[1, N], f32>, !mo.tensor<[1, N], si64>)
  mo.output %v, %i : !mo.tensor<[1, N], f32>, !mo.tensor<[1, N], si64>
}
