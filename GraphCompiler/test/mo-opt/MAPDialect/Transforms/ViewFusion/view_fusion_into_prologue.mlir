// RUN: mo-opt --mlir-print-local-scope %s --map-fuse-views | FileCheck %s

// Hand-authored MAP-level IR (mirrors
// PrologueFusion/prologue_fusion_map_ir.mlir): exercises ViewFuser's own
// fusion of a view directly into an existing map.iter.prologue, the one edge
// PrologueFuser's simpler mechanism can't take on its own (it splices a
// producer's body in verbatim, which a transform op's ParentOneOf rejects
// inside a prologue).

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.consumer registration = "kernels.consumer" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @sliceIntoExistingPrologue
// CHECK-SAME:  (%[[IN:.*]]: !mo.tensor<[10, 64], f32>, %[[STARTS:.*]]: !mo.tensor<[2], si64>, %[[STOPS:.*]]: !mo.tensor<[2], si64>, %[[STEPS:.*]]: !mo.tensor<[2], si64>)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       %[[RES:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[IN]]: !mo.tensor<[10, 64], f32>, %[[STARTS]]: !mo.tensor<[2], si64>, %[[STOPS]]: !mo.tensor<[2], si64>, %[[STEPS]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:  outputs (%[[O:.*]] : !mo.tensor<[3, 32], f32>) {
// CHECK-NEXT:  %[[PRO:.*]] = map.iter.prologue (%[[PI:.*]] : !map.index<2>) capturing
// CHECK-SAME:      {%[[CIN:.*]] = %[[IN]] : !mo.tensor<[10, 64], f32>,
// CHECK-SAME:       %[[CSTARTS:.*]] = %[[STARTS]] : !mo.tensor<[2], si64>,
// CHECK-SAME:       %[[CSTOPS:.*]] = %[[STOPS]] : !mo.tensor<[2], si64>,
// CHECK-SAME:       %[[CSTEPS:.*]] = %[[STEPS]] : !mo.tensor<[2], si64>}
// CHECK-NEXT:  %[[SIDX:.*]] = map.index.slice<{{.*}}> (%[[PI]], %[[CSTARTS]], %[[CSTOPS]], %[[CSTEPS]])
// CHECK-NEXT:  %[[VAL:.*]] = map.load %[[CIN]][%[[SIDX]]] : !mo.tensor<[10, 64], f32>
// CHECK-NEXT:  map.yield %[[VAL]] : f32
// CHECK-NEXT:  } : !mo.tensor<[3, 32], f32>
// CHECK-NEXT:  map.call.opaque @kernels.consumer[] inputs(%[[PRO]]) outputs(%[[O]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_prologue<[{{.*}}], operandIndices = [0]>>
// CHECK:       mo.output %[[RES]] : !mo.tensor<[3, 32], f32>
mo.graph @sliceIntoExistingPrologue(
    %input: !mo.tensor<[10, 64], f32>,
    %starts: !mo.tensor<[2], si64>,
    %stops: !mo.tensor<[2], si64>,
    %steps: !mo.tensor<[2], si64>
) -> !mo.tensor<[3, 32], f32> {
  %ew = map.iter.elementwise<[3, 32]>
      inputs  (%input: !mo.tensor<[10, 64], f32>, %starts: !mo.tensor<[2], si64>,
               %stops: !mo.tensor<[2], si64>, %steps: !mo.tensor<[2], si64>)
      outputs (%ewout: !mo.tensor<[3, 32], f32>)
      index   (%i: !map.index<2>)
      -> !mo.tensor<[3, 32], f32>
  {
    %si = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%i, %starts, %stops, %steps)
        : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
    %val = map.load %input[%si] : !mo.tensor<[10, 64], f32>
    map.store %ewout[%i] = %val : !mo.tensor<[3, 32], f32>
  }
  %res = map.iter.opaque inputs (%ew : !mo.tensor<[3, 32], f32>) outputs (%o : !mo.tensor<[3, 32], f32>) {
    %pro = map.iter.prologue (%pi : !map.index<2>) capturing {%pc = %ew : !mo.tensor<[3, 32], f32>} {
      %pv = map.load %pc[%pi] : !mo.tensor<[3, 32], f32>
      map.yield %pv : f32
    } : !mo.tensor<[3, 32], f32>
    map.call.opaque @kernels.consumer[] inputs (%pro) outputs (%o) kernelName = "kernels.consumer" : (!mo.tensor<[3, 32], f32>) -> (!mo.tensor<[3, 32], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[3, 32], f32>
}

// A plain elementwise (non-view) producer feeding an existing prologue is
// PrologueFuser's own job -- ViewFuser must leave it alone, keeping the two
// fusers' jobs disjoint. Negative control for the case above: same shape of
// graph, but the producer has no IndexTransformOpInterface op in its body.

// CHECK-LABEL: mo.graph @elementwiseIntoExistingPrologue_noFuse
// CHECK:       %[[EW:.*]] = map.iter.elementwise<[4]>
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[EW]]: !mo.tensor<[4], f32>)
// CHECK:       map.iter.prologue
// CHECK:       map.load
// CHECK-NOT:   map.index.slice
// CHECK:       map.yield
mo.graph @elementwiseIntoExistingPrologue_noFuse(
    %a: !mo.tensor<[4], f32>,
    %b: !mo.tensor<[4], f32>
) -> !mo.tensor<[4], f32> {
  %ew = map.iter.elementwise<[4]>
      inputs  (%a: !mo.tensor<[4], f32>, %b: !mo.tensor<[4], f32>)
      outputs (%ewout: !mo.tensor<[4], f32>)
      index   (%i: !map.index<1>)
      -> !mo.tensor<[4], f32>
  {
    %av = map.load %a[%i] : !mo.tensor<[4], f32>
    %bv = map.load %b[%i] : !mo.tensor<[4], f32>
    %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%i, %av, %bv)
               : !map.index<1>, f32, f32 -> f32
    map.store %ewout[%i] = %sum : !mo.tensor<[4], f32>
  }
  %res = map.iter.opaque inputs (%ew : !mo.tensor<[4], f32>) outputs (%o : !mo.tensor<[4], f32>) {
    %pro = map.iter.prologue (%pi : !map.index<1>) capturing {%pc = %ew : !mo.tensor<[4], f32>} {
      %pv = map.load %pc[%pi] : !mo.tensor<[4], f32>
      map.yield %pv : f32
    } : !mo.tensor<[4], f32>
    map.call.opaque @kernels.consumer[] inputs (%pro) outputs (%o) kernelName = "kernels.consumer" : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[4], f32>
}

}
