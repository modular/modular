// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s

// A concat returned whole and as a trailing window, through the full fusion
// pipeline: horizontal fusion puts the window's store in a select beside the
// concat's own, and the epilogue lowers to a plain store plus a
// `mogg.index.slice.select` holding the window's store at the select's index.
// The negative start stays as written, and the window is dynamic, so the select
// reads the slice's runtime in- and out-shapes, threaded into the epilogue as
// shape captures. The kernel still binds the concat output as the storage it
// writes; the window is only ever stored through, so the call must report
// epilogue fusion: a kernel told there is none writes its result itself and
// never runs the epilogue.

// CHECK-LABEL: mo.graph @concat_slice_window
// CHECK:       dims (%[[B:[a-z0-9]+]]: !mogg.dim<B>, %{{[a-z0-9]+}}: !mogg.dim<S>, %[[T:[a-z0-9]+]]: !mogg.dim<T>)
// CHECK:       %[[CONCAT:.*]] = mogg.tensor.create(%{{.*}}) {{.*}}shapeAttr = #mosh<ape[B, T, 16]>
// CHECK:       %[[WINDOW:.*]] = mogg.tensor.create(%{{.*}}) {{.*}}shapeAttr = #mosh<ape[B, 8, 16]>
// CHECK:       %[[IN_SHAPE:[0-9]+]] = mogg.shape.create(%[[B]], %[[T]]) <shape = #mosh<ape[dyn, dyn, 16]> : !mosh.ape>
// CHECK-NEXT:  %[[OUT_SHAPE:[0-9]+]] = mogg.shape.create(%[[B]]) <shape = #mosh<ape[dyn, 8, 16]> : !mosh.ape>
// CHECK:       %[[PTR:.*]] = mogg.tensor.get_ptr(%[[CONCAT]])
// CHECK:       %[[EPI:[^ ,)]+]] = mogg.tensor.create.epilogue %{{.*}} (%[[VAL:[a-z0-9]+]] : !mogg.simd<f32>, %[[IDX:[a-z0-9]+]] : !map.index<3>) capturing {%[[CAP0:[a-z0-9]+]] = %[[CONCAT]] : !mogg.tensor, %[[CAP1:[a-z0-9]+]] = %[[WINDOW]] : !mogg.tensor} shapes {%[[IN:[a-z0-9]+]] = %[[IN_SHAPE]], %[[OUT:[a-z0-9]+]] = %[[OUT_SHAPE]]} ptr %[[PTR]]
// CHECK-NEXT:    mogg.tensor.store %[[CAP0]][%[[IDX]]] = %[[VAL]]
// CHECK-NEXT:    mogg.index.slice.select(%[[IDX]], %[[IN]], %[[OUT]]) <inShape = #mosh<ape[B, T, 16]> : !mosh.ape, outShape = #mosh<ape[B, 8, 16]> : !mosh.ape, staticStart = array<i64: 0, -8, 0>, staticStep = array<i64: 1, 1, 1>> : !map.index<3> {
// CHECK-NEXT:    ^bb0(%[[IOUT:[a-z0-9]+]]: !map.index<3>):
// CHECK-NEXT:      mogg.tensor.store %[[CAP1]][%[[IOUT]]] = %[[VAL]]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  mogg.call.opaque @{{.*}}.FusedVariadicInputs[axis = 1 : i64] inputs({{.*}}) outputs(%[[EPI]] : !mogg.tensor) context({{.*}}) hasEpilogueFusion = true
// CHECK-NOT:   mogg.routine
mo.graph @concat_slice_window<B, S, T -> OB, OT>(
  %cache: !mo.tensor<[B, 8, 16], f32>,
  %new: !mo.tensor<[B, S, 16], f32>,
  %stops: !mo.tensor<[3], si64>
) -> (!mo.tensor<[OB, OT, 16], f32>, !mo.tensor<[OB, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%cache, %new) : (!mo.tensor<[B, 8, 16], f32>, !mo.tensor<[B, S, 16], f32>) -> !mo.tensor<[B, T, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[B, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[B, 8, 16], f32>
  mo.output<B, T> %concat, %slice : !mo.tensor<[B, T, 16], f32>, !mo.tensor<[B, 8, 16], f32>
}

// Matmul-family kernels call the epilogue only when told it does work, and
// write their own output otherwise. The select is work: without the epilogue,
// nothing writes the window. The pass-through the predicate matches is a single
// store of the value the kernel hands in, so a store beside a select is not one.

// CHECK-LABEL: mo.graph @matmul_slice_step
// CHECK:       %[[EPI:.*]] = mogg.tensor.create.epilogue
// CHECK:       mogg.index.slice.select(%{{[a-z0-9]+}}) <inShape = #mosh<ape[8, 8]> : !mosh.ape, outShape = #mosh<ape[4, 8]> : !mosh.ape, staticStart = array<i64: 1, 0>, staticStep = array<i64: 2, 1>> : !map.index<2> {
// CHECK-NEXT:  ^bb0(%[[IOUT:[a-z0-9]+]]: !map.index<2>):
// CHECK-NEXT:    mogg.tensor.store %{{[a-z0-9]+}}[%[[IOUT]]]
// CHECK:       mogg.call.opaque @{{.*}}.MatmulCPU{{(\[.*\])?}} inputs({{.*}}) outputs(%[[EPI]] : !mogg.tensor) {{.*}}hasEpilogueFusion = true
mo.graph @matmul_slice_step(
  %a: !mo.tensor<[8, 4], f32>,
  %b: !mo.tensor<[4, 8], f32>
) -> (!mo.tensor<[8, 8], f32>, !mo.tensor<[4, 8], f32>) {
  %starts = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<8, 8> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[8, 4], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %slice = mo.slice(%mm, %starts, %stops, %steps) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 8], f32>
  mo.output %mm, %slice : !mo.tensor<[8, 8], f32>, !mo.tensor<[4, 8], f32>
}

// CHECK-LABEL: mo.graph @bmm_window
// CHECK:       %[[EPI:.*]] = mogg.tensor.create.epilogue
// CHECK:       mogg.index.slice.select(%{{[a-z0-9]+}}) <inShape = #mosh<ape[2, 16, 16]> : !mosh.ape, outShape = #mosh<ape[2, 4, 16]> : !mosh.ape, staticStart = array<i64: 0, 12, 0>, staticStep = array<i64: 1, 1, 1>> : !map.index<3> {
// CHECK-NEXT:  ^bb0(%[[IOUT:[a-z0-9]+]]: !map.index<3>):
// CHECK-NEXT:    mogg.tensor.store %{{[a-z0-9]+}}[%[[IOUT]]]
// CHECK:       mogg.call.opaque @{{.*}}.BatchMatmulCPU{{(\[.*\])?}} inputs({{.*}}) outputs(%[[EPI]] : !mogg.tensor) {{.*}}hasEpilogueFusion = true
mo.graph @bmm_window(
  %a: !mo.tensor<[2, 16, 8], f32>,
  %b: !mo.tensor<[2, 8, 16], f32>
) -> (!mo.tensor<[2, 16, 16], f32>, !mo.tensor<[2, 4, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -4, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<2, 16, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %bmm = mo.batch_matmul(%a, %b) : (!mo.tensor<[2, 16, 8], f32>, !mo.tensor<[2, 8, 16], f32>) -> !mo.tensor<[2, 16, 16], f32>
  %slice = mo.slice(%bmm, %starts, %stops, %steps) : (!mo.tensor<[2, 16, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 4, 16], f32>
  mo.output %bmm, %slice : !mo.tensor<[2, 16, 16], f32>, !mo.tensor<[2, 4, 16], f32>
}

// Convolution skips its epilogue the same way.

// CHECK-LABEL: mo.graph @conv_window
// CHECK:       %[[EPI:.*]] = mogg.tensor.create.epilogue
// CHECK:       mogg.index.slice.select(%{{[a-z0-9]+}}) <inShape = #mosh<ape[2, 3, 3, 4]> : !mosh.ape, outShape = #mosh<ape[2, 2, 3, 4]> : !mosh.ape, staticStart = array<i64: 0, 1, 0, 0>, staticStep = array<i64: 1, 1, 1, 1>> : !map.index<4> {
// CHECK-NEXT:  ^bb0(%[[IOUT:[a-z0-9]+]]: !map.index<4>):
// CHECK-NEXT:    mogg.tensor.store %{{[a-z0-9]+}}[%[[IOUT]]]
// CHECK:       mogg.call.opaque @{{.*}}.ConvCPU{{(\[.*\])?}} inputs({{.*}}) outputs(%[[EPI]] : !mogg.tensor) {{.*}}hasEpilogueFusion = true
mo.graph @conv_window(
  %input: !mo.tensor<[2, 5, 5, 3], f32>,
  %filter: !mo.tensor<[3, 3, 3, 4], f32, {layout = #mo.layout<RSCF>}>
) -> (!mo.tensor<[2, 3, 3, 4], f32>, !mo.tensor<[2, 2, 3, 4], f32>) {
  %strides = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %paddings = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %starts = mo.constant <#M.dense_array<0, -2, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %stops = mo.constant <#M.dense_array<2, 3, 3, 4> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %conv = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %paddings, num_groups = %groups, filter_layout = ""] : (
    !mo.tensor<[2, 5, 5, 3], f32>, !mo.tensor<[3, 3, 3, 4], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 3, 3, 4], f32>
  %slice = mo.slice(%conv, %starts, %stops, %steps) : (!mo.tensor<[2, 3, 3, 4], f32>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[2, 2, 3, 4], f32>
  mo.output %conv, %slice : !mo.tensor<[2, 3, 3, 4], f32>, !mo.tensor<[2, 2, 3, 4], f32>
}

// Only the in-shape is dynamic: the select still reads both shapes at
// runtime.

// CHECK-LABEL: mo.graph @concat_window_dynamic_in
// CHECK:       mogg.tensor.create.epilogue %{{.*}} (%[[VAL:[a-z0-9]+]] : !mogg.simd<f32>, %[[IDX:[a-z0-9]+]] : !map.index<3>) capturing {%{{[a-z0-9]+}} = %{{.*}} : !mogg.tensor, %[[CAP1:[a-z0-9]+]] = %{{.*}} : !mogg.tensor} shapes {%[[IN:[a-z0-9]+]] = %{{[0-9]+}}, %[[OUT:[a-z0-9]+]] = %{{[0-9]+}}}
// CHECK-NEXT:    mogg.tensor.store
// CHECK-NEXT:    mogg.index.slice.select(%[[IDX]], %[[IN]], %[[OUT]]) <inShape = #mosh<ape[2, T, 16]> : !mosh.ape, outShape = #mosh<ape[2, 8, 16]> : !mosh.ape, staticStart = array<i64: 0, -8, 0>, staticStep = array<i64: 1, 1, 1>>
// CHECK-NEXT:    ^bb0(%[[IOUT:[a-z0-9]+]]: !map.index<3>):
// CHECK-NEXT:      mogg.tensor.store %[[CAP1]][%[[IOUT]]] = %[[VAL]]
mo.graph @concat_window_dynamic_in<S, T>(
  %cache: !mo.tensor<[2, 8, 16], f32>,
  %new: !mo.tensor<[2, S, 16], f32>,
  %stops: !mo.tensor<[3], si64>
) -> (!mo.tensor<[2, T, 16], f32>, !mo.tensor<[2, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, -8, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %concat = mo.custom [symbol = "fused_variadic_inputs", device = #M.device_ref<"cpu", 0>, parameters = {axis = 1}](%cache, %new) : (!mo.tensor<[2, 8, 16], f32>, !mo.tensor<[2, S, 16], f32>) -> !mo.tensor<[2, T, 16], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[2, T, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 8, 16], f32>
  mo.output %concat, %slice : !mo.tensor<[2, T, 16], f32>, !mo.tensor<[2, 8, 16], f32>
}

// A strided slice over a dynamic dim: both shapes are dynamic, the out-shape
// through the slice's own result dim.

// CHECK-LABEL: mo.graph @matmul_slice_step_dynamic
// CHECK:       mogg.tensor.create.epilogue %{{.*}} (%{{[a-z0-9]+}} : {{[^,]+}}, %[[IDX:[a-z0-9]+]] : !map.index<2>) capturing {{[{][^}]*[}]}} shapes {%[[IN:[a-z0-9]+]] = %{{[0-9]+}}, %[[OUT:[a-z0-9]+]] = %{{[0-9]+}}}
// CHECK-NEXT:    mogg.tensor.store
// CHECK-NEXT:    mogg.index.slice.select(%[[IDX]], %[[IN]], %[[OUT]]) <inShape = #mosh<ape[M, 8]> : !mosh.ape, outShape = #mosh<ape[H, 8]> : !mosh.ape, staticStart = array<i64: 1, 0>, staticStep = array<i64: 2, 1>>
// CHECK:       mogg.call.opaque @{{.*}}.MatmulCPU{{(\[.*\])?}} inputs({{.*}}) outputs({{.*}}) {{.*}}hasEpilogueFusion = true
mo.graph @matmul_slice_step_dynamic<M, H>(
  %a: !mo.tensor<[M, 4], f32>,
  %b: !mo.tensor<[4, 8], f32>,
  %stops: !mo.tensor<[2], si64>
) -> (!mo.tensor<[M, 8], f32>, !mo.tensor<[H, 8], f32>) {
  %starts = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mm = mo.matmul(%a, %b) : (!mo.tensor<[M, 4], f32>, !mo.tensor<[4, 8], f32>) -> !mo.tensor<[M, 8], f32>
  %slice = mo.slice(%mm, %starts, %stops, %steps) : (!mo.tensor<[M, 8], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[H, 8], f32>
  mo.output %mm, %slice : !mo.tensor<[M, 8], f32>, !mo.tensor<[H, 8], f32>
}
