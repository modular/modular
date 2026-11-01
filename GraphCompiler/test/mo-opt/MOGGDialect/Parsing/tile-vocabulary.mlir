// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trips the tile fusion vocabulary: the tile types and the tile
// load/store ops. The containers that hold them are
// covered by their own tests.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false

// CHECK-LABEL: func.func @types
// CHECK-SAME: (%{{.*}}: !mogg.tile<f32>, %{{.*}}: !mogg.tile<bf16>, %{{.*}}: !mogg.tile<si8>, %{{.*}}: !mogg.tile_coord<1>, %{{.*}}: !mogg.tile_coord<2>)
func.func @types(%a: !mogg.tile<f32>, %b: !mogg.tile<bf16>, %c: !mogg.tile<si8>, %coord1: !mogg.tile_coord<1>, %coord2: !mogg.tile_coord<2>) {
  return
}

// A capture is a block argument, so its element type is not known here.
// CHECK-LABEL: func.func @load_compute_store
// CHECK-SAME: (%[[OUT:.*]]: !mogg.tensor, %[[X:.*]]: !mogg.tensor, %[[COORD:.*]]: !mogg.tile_coord<2>)
func.func @load_compute_store(%out: !mogg.tensor, %x: !mogg.tensor, %coord: !mogg.tile_coord<2>) {
  // CHECK-NEXT: %[[XT:.*]] = mogg.tile.load %[[X]][%[[COORD]]] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
  %xt = mogg.tile.load %x[%coord] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
  // CHECK-NEXT: %[[CST:.*]] = mogg.constant.scalar
  %cst = mogg.constant.scalar 2.000000e+00 : f32 : f32
  // CHECK-NEXT: %[[SUM:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Add(%[[XT]], %[[CST]]) moOpName = "mo.add" : !mogg.tile<f32>, f32 -> !mogg.tile<f32>
  %sum = mogg.call.elementwise @builtin_kernels.elementwise.Add(%xt, %cst) moOpName = "mo.add" : !mogg.tile<f32>, f32 -> !mogg.tile<f32>
  // CHECK-NEXT: mogg.tile.store %[[OUT]][%[[COORD]]] = %[[SUM]] : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  mogg.tile.store %out[%coord] = %sum : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  return
}

// The element type is checked against a creating op in view.
// CHECK-LABEL: func.func @created_tensor
func.func @created_tensor(%p: !mogg.pointer, %coord: !mogg.tile_coord<2>) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %t = mogg.tensor.create(%p, %s) <device = "gpu", elementType = f16, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // CHECK: mogg.tile.load %{{.*}}[%{{.*}}] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f16>
  %v = mogg.tile.load %t[%coord] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f16>
  // CHECK: mogg.tile.store %{{.*}}[%{{.*}}] = %{{.*}} : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f16>
  mogg.tile.store %t[%coord] = %v : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f16>
  return
}

}
