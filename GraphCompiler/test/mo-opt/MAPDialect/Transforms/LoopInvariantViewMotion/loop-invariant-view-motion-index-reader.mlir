// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map-livm))' -verify-diagnostics

// Hoisting a view gives every kernel fused under it the loop's own index. That
// is wrong for an index reader, so the pass fails instead. Input fusion never
// builds this IR.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#IndexList = #lit<symbol@std::@utils::@index::@IndexList>
#SIMD = #lit<symbol@std::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

module {

// The trailing `IndexList` parameter makes the kernel an index reader.
mogg.kernel.decl @testing_kernels.testing_kernels.ElemwiseAddWithIndex registration = "elemwise_add_with_index" elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength, "rank": !Int>("input": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "indices": !lit.struct<#IndexList <:!Int *(0,2), :!DType {:dtype si64}>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>, #kgen<param.decl rank : !Int>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {element_type = #lit.struct<{_mlir_value: dtype = si64}> : !DType, size = #kgen.param.decl.ref<"rank"> : !Int}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = true

mo.graph @indexReaderUnderSlice(%input: !mo.tensor<[8], si32>,
                                %starts: !mo.tensor<[1], si64>,
                                %stops: !mo.tensor<[1], si64>,
                                %steps: !mo.tensor<[1], si64>)
    -> !mo.tensor<[4], si32> {
  %output = map.iter.opaque
      inputs (%input : !mo.tensor<[8], si32>, %starts : !mo.tensor<[1], si64>, %stops : !mo.tensor<[1], si64>, %steps : !mo.tensor<[1], si64>)
      outputs (%out : !mo.tensor<[4], si32>) {
    map.iter.elementwise.body<[4]>
        index (%idx: !map.index<1>)
        capturing {%input0 = %input : !mo.tensor<[8], si32>, %starts0 = %starts : !mo.tensor<[1], si64>, %stops0 = %stops : !mo.tensor<[1], si64>, %steps0 = %steps : !mo.tensor<[1], si64>, %out0 = %out : !mo.tensor<[4], si32>} {
      %idx_ = map.index.slice<#map.index.slice.spec<[8] -> [4], starts = [4], stops = [8], steps = [1]>> (%idx, %starts0, %stops0, %steps0)
          : (!map.index<1>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !map.index<1>
      %val = map.load %input0[%idx_] : !mo.tensor<[8], si32>
      // expected-error@+1 {{reads its element index but is fused under a view}}
      %res = map.call.elementwise @testing_kernels.testing_kernels.ElemwiseAddWithIndex(%idx_, %val) : !map.index<1>, si32 -> si32
      map.store %out0[%idx] = %res : !mo.tensor<[4], si32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %output : !mo.tensor<[4], si32>
}

}
