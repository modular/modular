// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// A dynamic axis reads its length from the shape. The fused square is part of
// the fold; the bias load and add run per row, at the emit's width.
// CHECK-LABEL: def __mul_reduce_add_add(
// CHECK:       var [[ROW2:.*]] = algorithm.rowwise.Row[params, DType.float32, DType.float32, 1, {{.*}}](row_coords, Int({{.*}}[1].value()), row_ctx, {{.*}})
// CHECK-NEXT:  def {{contribute_+[0-9]+}}[w : Int](tile: SIMD[DType.float32, w], {{.*}}) {var} -> SIMD[DType.float32, w]:
// CHECK-NEXT:    var [[SQ:.*]] = builtin_kernels.elementwise.Mul.elementwise[{{.*}}](tile, tile)
// CHECK-NEXT:    return [[SQ]]
// CHECK:       def {{write_+[0-9]+}}(oc: {{.*}}) {var}:
// CHECK-NEXT:    var [[B:.*]] = managed_tensor_slice.simd_load_from_tensor_pointer[{{.*}}simd_width = params.emit_tile_width{{.*}}oc.coord
// CHECK-NEXT:    var [[Y:.*]] = builtin_kernels.elementwise.Add.elementwise[{{.*}}]({{.*}}.acc.slice[params.emit_tile_width](), [[B]])
// CHECK-NEXT:    managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}[[Y]]))
mo.graph @sumOfSquaresPlusBias<M, N>(%x: !mo.tensor<[M, N], f32>, %b: !mo.tensor<[M, 1], f32>) -> !mo.tensor<[M, 1], f32> {
  %sq = mo.mul(%x, %x) : !mo.tensor<[M, N], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[M, N], f32>) -> !mo.tensor<[M, 1], f32>
  %y = mo.add(%r, %b) : !mo.tensor<[M, 1], f32>
  mo.output %y : !mo.tensor<[M, 1], f32>
}
