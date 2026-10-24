// RUN: mo-opt --mlir-print-local-scope %s --map-fuse-mutable-loads | FileCheck %s

// map.call.opaque verifies that its callee resolves to a mogg.kernel.decl. The
// pass never reads the signature, so an empty declaration is enough here.
mogg.kernel.decl @builtin_kernels.kernels.Struct_sampler_apply_penalties registration = "sampler.apply_penalties" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// Written as MAP IR rather than lowered from an mo.custom: MOToMAP only builds a
// map.iter.prologue for an input the kernel declares fusible, and no chained
// opaque kernel in the default kernel set declares one.
//
// Fusing the load retypes the kernel's input block argument from tensor to
// buffer. Where the kernel reads that input through a map.iter.prologue, the
// capture's own inner block argument mirrors the captured type and has to
// follow, so that the map.load inside the prologue reads the buffer.
// CHECK-LABEL: mo.graph @fuse_into_opaque_with_prologue(
// CHECK-SAME:  %[[BUF:.*]]: !mo.buffer<[3, 2], si32>, %[[TARGET:.*]]: !mo.buffer<[3, 2], si32>
// CHECK-NOT:   mo.mutable.load
// CHECK:       %[[CH0:.*]] = mo.chain.create()
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs (%[[TARGET]]: !mo.buffer<[3, 2], si32>, %[[BUF]]: !mo.buffer<[3, 2], si32>,
// CHECK-SAME:  %[[CH0]]: !mo.chain)
// CHECK:       %[[PRO:.*]] = map.iter.prologue (%[[IDX:.*]] : !map.index<2>) capturing {%[[CAP:.*]] = %[[BUF]] : !mo.buffer<[3, 2], si32>}
// CHECK:       map.load %[[CAP]][%[[IDX]]] : !mo.buffer<[3, 2], si32>
// CHECK:       map.call.opaque @{{.*}}sampler_apply_penalties[] inputs({{.*}}%[[PRO]],
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_mutable_load<[#mogg.fusion_leaf<"mo.mutable.load"{{.*}}, #mogg.fusion_leaf<"custom__sampler.apply_penalties"{{.*}}]>>
mo.graph @fuse_into_opaque_with_prologue(%buf: !mo.buffer<[3, 2], si32>, %target: !mo.buffer<[3, 2], si32>, %offsets: !mo.tensor<[11], ui32>, %penalty: !mo.tensor<[7], f32>) -> !mo.chain {
  %ch0 = mo.chain.create()
  %t, %oc = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[3, 2], si32>, !mo.chain) -> (!mo.tensor<[3, 2], si32>, !mo.chain)
  %c = map.iter.opaque
      inputs (%target : !mo.buffer<[3, 2], si32>, %t : !mo.tensor<[3, 2], si32>, %offsets : !mo.tensor<[11], ui32>, %penalty : !mo.tensor<[7], f32>, %ch0 : !mo.chain)
      outputs (%c : !mo.chain)
  {
    %pro = map.iter.prologue (%idx : !map.index<2>) capturing {%cap = %t : !mo.tensor<[3, 2], si32>} {
      %v = map.load %cap[%idx] : !mo.tensor<[3, 2], si32>
      map.yield %v : si32
    } : !mo.tensor<[3, 2], si32>
    map.call.opaque @builtin_kernels.kernels.Struct_sampler_apply_penalties[] inputs(%target, %pro, %offsets, %penalty, %penalty, %penalty) outputs() kernelName = "sampler.apply_penalties" : (!mo.buffer<[3, 2], si32>, !mo.tensor<[3, 2], si32>, !mo.tensor<[11], ui32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>, !mo.tensor<[7], f32>) -> ()
  } {device = #M.device_ref<"cpu", 0>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"custom__sampler.apply_penalties">>, hasReadEffect = true, hasWriteEffect = true}
  mo.output %c : !mo.chain
}
