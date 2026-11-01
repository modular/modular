// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map-livm, map.iter.opaque(canonicalize,cse)))' | FileCheck %s

// LoopInvariantViewMotion walks every CapturingFusionClosure generically, not
// just map.iter.elementwise.body -- this covers map.iter.prologue, whose own
// IsolatedFromAbove trait is exactly why the old direct-value-substitution
// hoisting approach was invalid there (see the pass's addCapture-based
// design). Hoists a map.index.slice chain feeding a map.load inside a
// map.iter.prologue's body into a map.index.slice.stride op in the enclosing
// map.iter.opaque, rewiring the load to a fresh capture of the hoisted view,
// same mechanic already proven for map.iter.elementwise.body in
// loop-invariant-view-motion-slice.mlir.

// map.call.opaque verifies that its callee resolves to a mogg.kernel.decl. The
// pass never reads the signature, so an empty declaration is enough here.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @prologueSliceHoist
// CHECK:         [[RESULT:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[INPUT:%[^ :,]+]]: !mo.tensor<[10, 64], f32>, [[STARTS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STOPS:%[^ :,]+]]: !mo.tensor<[2], si64>, [[STEPS:%[^ :,]+]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 32], f32>) {
// CHECK-NEXT:      [[STRIDED:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> ([[INPUT]], [[STARTS]], [[STOPS]], [[STEPS]])
// CHECK-NEXT:      [[PRO:%[^ ]+]] = map.iter.prologue ([[PIDX:%[^ :]+]] : !map.index<2>) capturing {[[NEWCAP:%[^ ]+]] = [[STRIDED]] : !mo.tensor<[3, 32], f32>} {
// CHECK-NOT:       map.index.slice
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[NEWCAP]]{{\[}}[[PIDX]]] : !mo.tensor<[3, 32], f32>
// CHECK-NEXT:      map.yield [[VAL]] : f32
// CHECK-NEXT:    } : !mo.tensor<[3, 32], f32>
// CHECK-NEXT:    map.call.opaque @kernels.relu[] inputs([[PRO]]) outputs([[OUT]])
mo.graph @prologueSliceHoist(%input: !mo.tensor<[10, 64], f32>,
                            %starts: !mo.tensor<[2], si64>,
                            %stops: !mo.tensor<[2], si64>,
                            %steps: !mo.tensor<[2], si64>)
    -> !mo.tensor<[3, 32], f32> {
  %res = map.iter.opaque
      inputs (%input : !mo.tensor<[10, 64], f32>, %starts : !mo.tensor<[2], si64>, %stops : !mo.tensor<[2], si64>, %steps : !mo.tensor<[2], si64>)
      outputs (%out : !mo.tensor<[3, 32], f32>) {
    %pro = map.iter.prologue (%pidx : !map.index<2>)
        capturing {%input0 = %input : !mo.tensor<[10, 64], f32>, %starts0 = %starts : !mo.tensor<[2], si64>, %stops0 = %stops : !mo.tensor<[2], si64>, %steps0 = %steps : !mo.tensor<[2], si64>} {
      %idx_ = map.index.slice<#map.index.slice.spec<[10, 64] -> [3, 32], starts = [1, 32], stops = [4, 64], steps = [1, 1]>> (%pidx, %starts0, %stops0, %steps0)
          : (!map.index<2>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !map.index<2>
      %v = map.load %input0[%idx_] : !mo.tensor<[10, 64], f32>
      map.yield %v : f32
    } : !mo.tensor<[3, 32], f32>
    map.call.opaque @kernels.relu[] inputs (%pro) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 32], f32>) -> (!mo.tensor<[3, 32], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[3, 32], f32>
}
