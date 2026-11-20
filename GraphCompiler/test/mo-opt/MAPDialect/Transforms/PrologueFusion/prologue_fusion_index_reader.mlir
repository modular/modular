// RUN: mo-opt --mlir-print-local-scope %s --map-fuse-prologue | FileCheck %s

// An index reader does not fuse into a prologue that reads it through a slice.
// Fused, it would see the consumer's index.

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#IndexList = #lit<symbol@std::@utils::@index::@IndexList>
#SIMD = #lit<symbol@std::@simd::@SIMD>
!Int = !lit.struct<#SIMD <:!DType {:dtype index}, :!SIMDLength {1}>>

module {

mogg.kernel.decl @testing_kernels.testing_kernels.ElemwiseAddWithIndex registration = "elemwise_add_with_index" elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength, "rank": !Int>("input": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "indices": !lit.struct<#IndexList <:!Int *(0,2), :!DType {:dtype si64}>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>, #kgen<param.decl rank : !Int>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {element_type = #lit.struct<{_mlir_value: dtype = si64}> : !DType, size = #kgen.param.decl.ref<"rank"> : !Int}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = true
mogg.kernel.decl @kernels.consumer registration = "kernels.consumer" struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @indexReaderThroughSliceInPrologue
// CHECK:       map.iter.elementwise<[8]>
// CHECK:       map.call.elementwise @testing_kernels.testing_kernels.ElemwiseAddWithIndex
// CHECK:       map.iter.prologue
// CHECK-NOT:   ElemwiseAddWithIndex
// CHECK:       map.call.opaque @kernels.consumer
mo.graph @indexReaderThroughSliceInPrologue(
    %x: !mo.tensor<[8], si32>,
    %starts: !mo.tensor<[1], si64>,
    %stops: !mo.tensor<[1], si64>,
    %steps: !mo.tensor<[1], si64>
) -> !mo.tensor<[4], si32> {
  %z = map.iter.elementwise<[8]>
      inputs  (%x: !mo.tensor<[8], si32>)
      outputs (%zout: !mo.tensor<[8], si32>)
      index   (%i: !map.index<1>)
      -> !mo.tensor<[8], si32>
  {
    %xv = map.load %x[%i] : !mo.tensor<[8], si32>
    %zv = map.call.elementwise @testing_kernels.testing_kernels.ElemwiseAddWithIndex(%i, %xv)
              : !map.index<1>, si32 -> si32
    map.store %zout[%i] = %zv : !mo.tensor<[8], si32>
  }
  %res = map.iter.opaque inputs (%z : !mo.tensor<[8], si32>, %starts : !mo.tensor<[1], si64>, %stops : !mo.tensor<[1], si64>, %steps : !mo.tensor<[1], si64>) outputs (%o : !mo.tensor<[4], si32>) {
    %pro = map.iter.prologue (%pi : !map.index<1>) capturing {%pc = %z : !mo.tensor<[8], si32>, %cstarts = %starts : !mo.tensor<[1], si64>, %cstops = %stops : !mo.tensor<[1], si64>, %csteps = %steps : !mo.tensor<[1], si64>} {
      %si = map.index.slice<#map.index.slice.spec<[8] -> [4], starts = [4], stops = [8], steps = [1]>> (%pi, %cstarts, %cstops, %csteps)
          : (!map.index<1>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !map.index<1>
      %pv = map.load %pc[%si] : !mo.tensor<[8], si32>
      map.yield %pv : si32
    } : !mo.tensor<[4], si32>
    map.call.opaque @kernels.consumer[] inputs (%pro) outputs (%o) kernelName = "kernels.consumer" : (!mo.tensor<[4], si32>) -> (!mo.tensor<[4], si32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[4], si32>
}

}
