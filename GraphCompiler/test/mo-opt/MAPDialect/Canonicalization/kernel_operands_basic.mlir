// RUN: mo-opt %s --canonicalize | FileCheck %s

mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.add registration = "kernels.add" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @drops_unused_input
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[USED:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK:         map.call.opaque @kernels.relu[] inputs([[USED]]) outputs
mo.graph @drops_unused_input(
  %used: !mo.tensor<[2, 3, 4], f32>,
  %unused: !mo.tensor<[2, 3, 4], f32>
) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%used : !mo.tensor<[2, 3, 4], f32>, %unused : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    map.call.opaque @kernels.relu[] inputs (%used) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %y : !mo.tensor<[2, 3, 4], f32>
}

// CHECK-LABEL: mo.graph @dedupes_repeated_input
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[A:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK:         map.call.opaque @kernels.add[] inputs([[A]], [[A]]) outputs
mo.graph @dedupes_repeated_input(
  %x: !mo.tensor<[2, 3, 4], f32>
) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%x as %a : !mo.tensor<[2, 3, 4], f32>, %x as %b : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    map.call.opaque @kernels.add[] inputs (%a, %b) outputs (%out) kernelName = "kernels.add" : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %y : !mo.tensor<[2, 3, 4], f32>
}

// CHECK-LABEL: mo.graph @keeps_prologue_captured_input
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[X:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>)
mo.graph @keeps_prologue_captured_input(
  %x: !mo.tensor<[2, 3, 4], f32>
) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%x : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    %pro = map.iter.prologue (%pidx : !map.index<3>) capturing {%pcap = %x : !mo.tensor<[2, 3, 4], f32>} {
      %pval = map.load %pcap[%pidx] : !mo.tensor<[2, 3, 4], f32>
      map.yield %pval : f32
    } : !mo.tensor<[2, 3, 4], f32>
    map.call.opaque @kernels.relu[] inputs (%pro) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %y : !mo.tensor<[2, 3, 4], f32>
}

// Capture dedup leaves the first load dead, and the canonicalizer's DCE
// erases it in the same fixed point -- only the yielded load survives.
// CHECK-LABEL: mo.graph @dedupes_prologue_captures
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[A:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK:         map.iter.prologue (%{{[^ :]+}} : !map.index<3>) capturing {[[CAP:%[^ =]+]] = [[A]] : !mo.tensor<[2, 3, 4], f32>} {
// CHECK-NEXT:      [[VAL:%[^ ]+]] = map.load [[CAP]]
// CHECK-NEXT:      map.yield [[VAL]]
mo.graph @dedupes_prologue_captures(
  %x: !mo.tensor<[2, 3, 4], f32>
) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%x as %a : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    %pro = map.iter.prologue (%pidx : !map.index<3>) capturing {%pcap0 = %a : !mo.tensor<[2, 3, 4], f32>, %pcap1 = %a : !mo.tensor<[2, 3, 4], f32>} {
      %pval0 = map.load %pcap0[%pidx] : !mo.tensor<[2, 3, 4], f32>
      %pval1 = map.load %pcap1[%pidx] : !mo.tensor<[2, 3, 4], f32>
      map.yield %pval1 : f32
    } : !mo.tensor<[2, 3, 4], f32>
    map.call.opaque @kernels.relu[] inputs (%pro) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %y : !mo.tensor<[2, 3, 4], f32>
}

// The duplicated captures sit behind the epilogue's leading value/index args.
// The duplicate is folded into one capture; its loads are dead after dedup
// and DCE'd. That leaves the surviving %a capture with no uses at all, so
// DropDeadCapturingClosureCaptures drops it too in the same fixed point --
// only the %out capture (still used by the store) survives.
// CHECK-LABEL: mo.graph @dedupes_epilogue_captures
// CHECK:       map.iter.opaque
// CHECK-NEXT:  inputs ([[A:%[^ :,]+]]: !mo.tensor<[2, 3, 4], f32>)
// CHECK-NEXT:  outputs ([[O:%[^ :,]+]] : !mo.tensor<[2, 3, 4], f32>)
// CHECK:         map.iter.epilogue ([[VAL:%[^ :]+]] : f32, %{{[^ :]+}} : !map.index<3>) capturing {[[OCAP:%[^ =]+]] = [[O]] : !mo.tensor<[2, 3, 4], f32>} {
// CHECK-NEXT:      map.store [[OCAP]][%{{[^]]+}}] = [[VAL]]
mo.graph @dedupes_epilogue_captures(
  %x: !mo.tensor<[2, 3, 4], f32>
) -> !mo.tensor<[2, 3, 4], f32> {
  %y = map.iter.opaque inputs (%x as %a : !mo.tensor<[2, 3, 4], f32>) outputs (%out : !mo.tensor<[2, 3, 4], f32>) {
    %epi = map.iter.epilogue (%eval : f32, %eidx : !map.index<3>) capturing {%ecap = %out : !mo.tensor<[2, 3, 4], f32>, %ecap0 = %a : !mo.tensor<[2, 3, 4], f32>, %ecap1 = %a : !mo.tensor<[2, 3, 4], f32>} {
      %bias0 = map.load %ecap0[%eidx] : !mo.tensor<[2, 3, 4], f32>
      %bias1 = map.load %ecap1[%eidx] : !mo.tensor<[2, 3, 4], f32>
      map.store %ecap[%eidx] = %eval : !mo.tensor<[2, 3, 4], f32>
    } : !mo.tensor<[2, 3, 4], f32>
    map.call.opaque @kernels.relu[] inputs (%a) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[2, 3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %y : !mo.tensor<[2, 3, 4], f32>
}
