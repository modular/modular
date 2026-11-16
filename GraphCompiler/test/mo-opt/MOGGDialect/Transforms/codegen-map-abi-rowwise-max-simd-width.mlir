// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages --map-to-mogg \
// RUN:   --mogg-legalize-rowwise-kernels --mogg-conform-call-operands \
// RUN:   --mogg-outline-routines --mogg-to-mgp --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s

// A reduction's width limit reaches `_rowwise` as its `max_simd_width`, as
// for `_foreach`: an `Idx` when static, else the run-time largest power of
// two dividing the limit's terms, static part included.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

// map.iter.reduce verifies its monoid resolves to a mogg.kernel.decl.
mogg.kernel.decl @builtin_kernels.monoids.ReduceSum registration = "mo.reduce.add" reduction = <<"dtype": !DType, "W": !Int>, "W", [#mogg.kernel_reduction_output<"acc", #kgen.param.decl.ref<"dtype"> : !DType>]> struct_params = [#kgen<param.decl dtype : !DType>, #kgen<param.decl W : !Int>] is_user_defined = false

// CHECK:      var [[DIV:var_+[0-9]+]] = builtin_primitives.primitives.pow2_divisor_of_dims(std.utils.coord.coord_to_index_list({{var_+[0-9]+}}))
// CHECK:      builtin_primitives.primitives._rowwise[{{.*}}]({{.*}}, max_simd_width = Scalar[DType.int]([[DIV]]))
mo.graph @dynamic<N>(%x: !mo.tensor<[N, 32], f32>) -> !mo.tensor<[N, 1], f32> {
  %s = map.iter.opaque
      inputs (%x : !mo.tensor<[N, 32], f32>)
      outputs (%sum : !mo.tensor<[N, 1], f32>) {
    map.iter.rowwise.body<[N, 32], axis = 1> max_simd_width(#map.gcd<[8, N]>)
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[N, 32], f32>, %sum0 = %sum : !mo.tensor<[N, 1], f32>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
        %v = map.load %x0[%i] : !mo.tensor<[N, 32], f32>
        map.yield %v : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[N, 1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %s : !mo.tensor<[N, 1], f32>
}

// CHECK:      builtin_primitives.primitives._rowwise[{{.*}}]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, max_simd_width = layout.Idx[2])
mo.graph @static(%x: !mo.tensor<[4, 32], f32>) -> !mo.tensor<[4, 1], f32> {
  %s = map.iter.opaque
      inputs (%x : !mo.tensor<[4, 32], f32>)
      outputs (%sum : !mo.tensor<[4, 1], f32>) {
    map.iter.rowwise.body<[4, 32], axis = 1> max_simd_width(#map.gcd<[6, 10]>)
        row (%row: !map.index<2>)
        capturing (%x0 = %x : !mo.tensor<[4, 32], f32>, %sum0 = %sum : !mo.tensor<[4, 1], f32>) {
      %acc = map.iter.reduce @builtin_kernels.monoids.ReduceSum index (%i: !map.index<2>) -> f32 {
        %v = map.load %x0[%i] : !mo.tensor<[4, 32], f32>
        map.yield %v : f32
      }
      map.store %sum0[%row] = %acc : !mo.tensor<[4, 1], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %s : !mo.tensor<[4, 1], f32>
}
