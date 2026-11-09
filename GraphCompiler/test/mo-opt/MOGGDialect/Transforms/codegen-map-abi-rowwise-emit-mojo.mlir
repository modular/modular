// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// A rowwise kernel prints as a Row API body: its cache as the `load` closure, a
// `Row`, one `row.reduce` per fold, one `row.elementwise` per map phase, and
// per-row stores in `row.emit`. The
// launch parameters come from the op, and `mogg.call.rowwise` picks the SIMD
// width.

// A static axis gives the Row a compile-time length.
// CHECK-LABEL: def __reduce_add(
// CHECK:       comptime [[SPEC_X:.*]] = [[MTS_X:.*]].static_spec
// CHECK:       comptime [[SPEC_O:.*]] = [[MTS_O:.*]].static_spec
// CHECK:       def [[BODY:body_+[0-9]+]][params : algorithm.rowwise_types.ContextParams](row_coords: layout.Coord, mut row_ctx: algorithm.rowwise_types.Context[params]) {var}:
// CHECK-NEXT:    comptime [[RANK:.*]] = row_coords.rank
// CHECK-NEXT:    def [[LOAD:load_+[0-9]+]][width : Int, alignment : Int](idx: std.utils.coord.DynamicCoord[DType.int, [[RANK]]]) {var} -> SIMD[DType.float32, width]:
// CHECK-NEXT:      var [[X:.*]] = managed_tensor_slice.simd_load_from_tensor_pointer[static_spec = [[SPEC_X]], simd_width = width, element_alignment = alignment]({{.*}}idx{{.*}})
// CHECK-NEXT:      return [[X]]
// CHECK:         var [[ROW:.*]] = algorithm.rowwise.Row[params, DType.float32, DType.float32, 1, [[RANK]], is_cached = False](row_coords, layout.Idx[128], row_ctx, [[LOAD]])
// CHECK-NEXT:    def [[ADD:contribute_+[0-9]+]][w : Int](tile: SIMD[DType.float32, w], idx: std.utils.coord.DynamicCoord[DType.int, [[RANK]]]) {var} -> SIMD[DType.float32, w]:
// CHECK-NEXT:      return tile
// CHECK:         var [[ACC:.*]] = [[ROW]].reduce[builtin_kernels.monoids.ReduceSum[dtype = DType.float32, W = params.simd_width]]([[ADD]], [[LOAD]])
// CHECK-NEXT:    def [[WRITE:write_+[0-9]+]](oc: std.utils.coord.DynamicCoord[DType.int, [[RANK]]]) {var}:
// CHECK-NEXT:      managed_tensor_slice.simd_store_into_tensor_pointer[static_spec = [[SPEC_O]], simd_width = params.emit_tile_width, element_alignment = 1]({{.*}}oc{{.*}}[[ACC]].acc.slice[params.emit_tile_width]()))
// CHECK:         [[ROW]].emit([[WRITE]])
// CHECK:       {{.*}}._rowwise[builtin_kernels.monoids.ReduceSum[dtype = DType.float32, W = 1], DType.float32, DType.float32, axis = 1, num_phases = 1, computationally_expensive = False, dtype_size = 0]([[BODY]], {{.*}})
mo.graph @sum(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// A bf16 input summed in f32: the Row and the SIMD width see both dtypes, and
// the cast runs in the fold.
// CHECK-LABEL: def __cast_reduce_add(
// CHECK:       algorithm.rowwise.Row[params, DType.float32, DType.bfloat16, 1,
// CHECK:       def {{contribute_+[0-9]+}}[w : Int](tile: SIMD[DType.bfloat16, w], idx: {{.*}}) {var} -> SIMD[DType.float32, w]:
// CHECK-NEXT:    {{.*}}Cast{{.*}}tile
// CHECK:       {{.*}}._rowwise[builtin_kernels.monoids.ReduceSum[dtype = DType.float32, W = 1], DType.bfloat16, DType.float32,
mo.graph @bf16Sum(%x: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 1], f32> {
  %a = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %r = mo.reduce.add(%a) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// The fused product is the cache, as with a fused input: the Row computes it
// once per element, with the constant at the `load` closure's width.
// CHECK-LABEL: def __static_broadcast_to_mul_{{[0-9a-f]+}}(
// CHECK:       def {{load_+[0-9]+}}[width : Int, alignment : Int](idx: {{.*}}) {var} -> SIMD[DType.float32, width]:
// CHECK-NEXT:    var [[V:.*]] = managed_tensor_slice.simd_load_from_tensor_pointer[{{.*}}simd_width = width, element_alignment = alignment]
// CHECK-NEXT:    var [[M:.*]] = builtin_kernels.elementwise.Mul.elementwise[dtype = DType.float32]([[V]], SIMD[DType.float32, width](2.0))
// CHECK-NEXT:    return [[M]]
// CHECK:       def {{contribute_+[0-9]+}}[w : Int](tile: SIMD[DType.float32, w], {{.*}}) {var} -> SIMD[DType.float32, w]:
// CHECK-NEXT:    return tile
mo.graph @scaledSum(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %c = mo.constant <#M.dense_array<2.0> : tensor<f32>> : !mo.tensor<[], f32>
  %cb = mo.static.broadcast_to(%c) : !mo.tensor<[], f32> -> !mo.tensor<[8, 128], f32>
  %m = mo.mul(%x, %cb) : !mo.tensor<[8, 128], f32>
  %r = mo.reduce.add(%m) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}
