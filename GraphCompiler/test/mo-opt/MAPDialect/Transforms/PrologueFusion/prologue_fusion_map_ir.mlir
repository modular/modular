// RUN: mo-opt --mlir-print-local-scope %s --map-fuse-prologue | FileCheck %s

!DType = !lit.struct<@std::@builtin::@dtype::@DType>
!SIMDLength = !lit.struct<@std::@builtin::@simd_length::@SIMDLength>
#SIMD = #lit<symbol@std::@builtin::@simd::@SIMD>

module {

mogg.kernel.decl @builtin_kernels.elementwise.Add registration = "mo.add" device_info = <"", "", "", ""> elementwise = <<!lit.generator<<"dtype": !DType, "width": !SIMDLength>("lhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>, "rhs": !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>) -> !lit.struct<#SIMD <:!DType *(0,0), :!SIMDLength *(0,1)>>>>, [#kgen<param.decl dtype : !DType>, #kgen<param.decl width : !SIMDLength>], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}, {dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}], [{dtype = #kgen.param.decl.ref<"dtype"> : !DType, size = #kgen.param.decl.ref<"width"> : !SIMDLength}]> struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.consumer registration = "kernels.consumer" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @fuse_add_into_prologue
// CHECK-SAME:  (%[[A:.*]]: !mo.tensor<[4], f32>, %[[B:.*]]: !mo.tensor<[4], f32>)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       %[[RES:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[A]]: !mo.tensor<[4], f32>, %[[B]]: !mo.tensor<[4], f32>)
// CHECK-NEXT:  outputs (%[[O:.*]] : !mo.tensor<[4], f32>) {
// CHECK-NEXT:  %[[PRO:.*]] = map.iter.prologue (%[[PI:.*]] : !map.index<1>) capturing {%[[CA:.*]] = %[[A]] : !mo.tensor<[4], f32>, %[[CB:.*]] = %[[B]] : !mo.tensor<[4], f32>}
// CHECK-NEXT:  %[[AV:.*]] = map.load %[[CA]][%[[PI]]] : !mo.tensor<[4], f32>
// CHECK-NEXT:  %[[BV:.*]] = map.load %[[CB]][%[[PI]]] : !mo.tensor<[4], f32>
// CHECK-NEXT:  %[[SUM:.*]] = map.call.elementwise @builtin_kernels.elementwise.Add(%[[PI]], %[[AV]], %[[BV]]) : !map.index<1>, f32, f32 -> f32
// CHECK-NEXT:  map.yield %[[SUM]] : f32
// CHECK-NEXT:  } : !mo.tensor<[4], f32>
// CHECK-NEXT:  map.call.opaque @kernels.consumer[] inputs(%[[PRO]]) outputs(%[[O]])
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_prologue<[{{.*}}], operandIndices = [0]>>
// CHECK:       mo.output %[[RES]] : !mo.tensor<[4], f32>
mo.graph @fuse_add_into_prologue(
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

// CHECK-LABEL: mo.graph @fuse_add_into_prologue_dynamic
// CHECK-SAME:  (%[[DA:.*]]: !mo.tensor<[D0], f32>, %[[DB:.*]]: !mo.tensor<[D0], f32>)
// CHECK-NOT:   map.iter.elementwise
// CHECK:       %[[DRES:.*]] = map.iter.opaque
// CHECK-NEXT:  inputs (%[[DA]]: !mo.tensor<[D0], f32>, %[[DB]]: !mo.tensor<[D0], f32>)
// CHECK:       map.iter.prologue (%[[DPI:.*]] : !map.index<1>) capturing {%[[DCA:.*]] = %[[DA]] : !mo.tensor<[D0], f32>, %[[DCB:.*]] = %[[DB]] : !mo.tensor<[D0], f32>}
// CHECK:       map.call.elementwise @builtin_kernels.elementwise.Add({{.*}}) : !map.index<1>, f32, f32 -> f32
// CHECK:       map.yield
// CHECK:       mo.output %[[DRES]] : !mo.tensor<[D0], f32>
mo.graph @fuse_add_into_prologue_dynamic<D0>(
    %a: !mo.tensor<[D0], f32>,
    %b: !mo.tensor<[D0], f32>
) -> !mo.tensor<[D0], f32> {
  %ew = map.iter.elementwise<[D0]>
      inputs  (%a: !mo.tensor<[D0], f32>, %b: !mo.tensor<[D0], f32>)
      outputs (%ewout: !mo.tensor<[D0], f32>)
      index   (%i: !map.index<1>)
      -> !mo.tensor<[D0], f32>
  {
    %av = map.load %a[%i] : !mo.tensor<[D0], f32>
    %bv = map.load %b[%i] : !mo.tensor<[D0], f32>
    %sum = map.call.elementwise @builtin_kernels.elementwise.Add(%i, %av, %bv)
               : !map.index<1>, f32, f32 -> f32
    map.store %ewout[%i] = %sum : !mo.tensor<[D0], f32>
  }
  %res = map.iter.opaque inputs (%ew : !mo.tensor<[D0], f32>) outputs (%o : !mo.tensor<[D0], f32>) {
    %pro = map.iter.prologue (%pi : !map.index<1>) capturing {%pc = %ew : !mo.tensor<[D0], f32>} {
      %pv = map.load %pc[%pi] : !mo.tensor<[D0], f32>
      map.yield %pv : f32
    } : !mo.tensor<[D0], f32>
    map.call.opaque @kernels.consumer[] inputs (%pro) outputs (%o) kernelName = "kernels.consumer" : (!mo.tensor<[D0], f32>) -> (!mo.tensor<[D0], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[D0], f32>
}

// A producer whose result is still needed elsewhere must stay materialized:
// the `hasOneUse` gate declines, so this is the negative control for the two
// cases above -- same shape, one extra use.

// CHECK-LABEL: mo.graph @no_fuse_multi_use
// CHECK:       %[[EW:.*]] = map.iter.elementwise<[4]>
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[EW]]: !mo.tensor<[4], f32>)
// CHECK:       map.load
// CHECK-NOT:   map.call.elementwise
// CHECK:       map.yield
mo.graph @no_fuse_multi_use(
    %a: !mo.tensor<[4], f32>,
    %b: !mo.tensor<[4], f32>
) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
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
  mo.output %res, %ew : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

}
