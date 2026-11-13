// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map-fuse-contiguous-views, canonicalize))' | FileCheck %s

// FuseContiguousViews rewrites a kernel input that a contiguous mogg.slice
// produces to read the slice's input through a map.index.slice.stride at the
// top of the kernel's body. The stride op takes rank-1 starts, stops and steps,
// built as host constants from the slice's constant bounds. Its spec leaves the
// start of the cut dimension dynamic, so the view reads it from the starts
// constant. The kernels here run on a device, so the canonicalizer keeps those
// constants as inputs rather than folding them into the body.

// map.call.opaque verifies that its callee resolves to a mogg.kernel.decl. The
// pass never reads the signature, so an empty declaration is enough here.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @sliceIntoCall
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK-DAG:     [[STARTS:%[^ ]+]] = mo.constant <#M.dense_array<1, 0> {{.*}}> : !mo.tensor<[2], si64>
// CHECK-DAG:     [[STOPS:%[^ ]+]] = mo.constant <#M.dense_array<4, 64> {{.*}}> : !mo.tensor<[2], si64>
// CHECK-DAG:     [[STEPS:%[^ ]+]] = mo.constant <#M.dense_array<1, 1> {{.*}}> : !mo.tensor<[2], si64>
// CHECK-NOT:     mogg.slice
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[10, 64], f32>, [[STARTS]]: !mo.tensor<[2], si64>, [[STOPS]]: !mo.tensor<[2], si64>, [[STEPS]]: !mo.tensor<[2], si64>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 64], f32>) {
// CHECK-NEXT:      [[VIEW:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 64], starts = [?, 0], stops = [4, 64], steps = [1, 1]>> ([[X]], [[STARTS]], [[STOPS]], [[STEPS]])
// CHECK-NEXT:      map.call.opaque @kernels.relu[] inputs([[VIEW]]) outputs([[OUT]])
mo.graph @sliceIntoCall(%x: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 64], f32> {
  %start = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %stop = mo.constant <#M.dense_array<4> : tensor<si64>> : !mo.tensor<[], si64>
  %s = mogg.slice(%x, %start, %stop) : (!mo.tensor<[10, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32>
  %res = map.iter.opaque
      inputs (%s : !mo.tensor<[3, 64], f32>)
      outputs (%out : !mo.tensor<[3, 64], f32>) {
    map.call.opaque @kernels.relu[] inputs (%s) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 64], f32>) -> (!mo.tensor<[3, 64], f32>)
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %res : !mo.tensor<[3, 64], f32>
}

// The slice cuts the first dim not of size 1, past any leading unit dims.

// CHECK-LABEL: mo.graph @sliceAfterUnitDims
// CHECK:           map.index.slice.stride<#map.index.slice.spec<[1, 10, 64] -> [1, 3, 64], starts = [0, ?, 0], stops = [1, 4, 64], steps = [1, 1, 1]>>
mo.graph @sliceAfterUnitDims(%x: !mo.tensor<[1, 10, 64], f32>) -> !mo.tensor<[1, 3, 64], f32> {
  %start = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %stop = mo.constant <#M.dense_array<4> : tensor<si64>> : !mo.tensor<[], si64>
  %s = mogg.slice(%x, %start, %stop) : (!mo.tensor<[1, 10, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1, 3, 64], f32>
  %res = map.iter.opaque
      inputs (%s : !mo.tensor<[1, 3, 64], f32>)
      outputs (%out : !mo.tensor<[1, 3, 64], f32>) {
    map.call.opaque @kernels.relu[] inputs (%s) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[1, 3, 64], f32>) -> (!mo.tensor<[1, 3, 64], f32>)
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %res : !mo.tensor<[1, 3, 64], f32>
}

// Kernels that slice one tensor at different rows get the same dynamic start;
// each reads its own from its starts constant.

// CHECK-LABEL: mo.graph @slicesOfOneTensor
// CHECK-DAG:     [[STARTS0:%[^ ]+]] = mo.constant <#M.dense_array<0, 0> {{.*}}> : !mo.tensor<[2], si64>
// CHECK-DAG:     [[STARTS1:%[^ ]+]] = mo.constant <#M.dense_array<3, 0> {{.*}}> : !mo.tensor<[2], si64>
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ({{.*}}, [[STARTS0]]: !mo.tensor<[2], si64>,
// CHECK-NEXT:    outputs
// CHECK-NEXT:      map.index.slice.stride<#map.index.slice.spec<[6, 64] -> [3, 64], starts = [?, 0],
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ({{.*}}, [[STARTS1]]: !mo.tensor<[2], si64>,
// CHECK-NEXT:    outputs
// CHECK-NEXT:      map.index.slice.stride<#map.index.slice.spec<[6, 64] -> [3, 64], starts = [?, 0],
mo.graph @slicesOfOneTensor(%x: !mo.tensor<[6, 64], f32>) -> (!mo.tensor<[3, 64], f32>, !mo.tensor<[3, 64], f32>) {
  %c0 = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
  %c3 = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>
  %c6 = mo.constant <#M.dense_array<6> : tensor<si64>> : !mo.tensor<[], si64>
  %s0 = mogg.slice(%x, %c0, %c3) : (!mo.tensor<[6, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32>
  %s1 = mogg.slice(%x, %c3, %c6) : (!mo.tensor<[6, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32>
  %r0 = map.iter.opaque
      inputs (%s0 : !mo.tensor<[3, 64], f32>)
      outputs (%out : !mo.tensor<[3, 64], f32>) {
    map.call.opaque @kernels.relu[] inputs (%s0) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 64], f32>) -> (!mo.tensor<[3, 64], f32>)
  } {device = #M.device_ref<"gpu", 0>}
  %r1 = map.iter.opaque
      inputs (%s1 : !mo.tensor<[3, 64], f32>)
      outputs (%out : !mo.tensor<[3, 64], f32>) {
    map.call.opaque @kernels.relu[] inputs (%s1) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 64], f32>) -> (!mo.tensor<[3, 64], f32>)
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %r0, %r1 : !mo.tensor<[3, 64], f32>, !mo.tensor<[3, 64], f32>
}

// A reshape of a slice fuses whole: the kernel reads the slice's input through
// a slice view, and the reshape view reads that.

// CHECK-LABEL: mo.graph @reshapeOfSlice
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK-NOT:     mogg.slice
// CHECK-NOT:     mo.static.reshape
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[10, 64], f32>, {{.*}})
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[192], f32>) {
// CHECK-NEXT:      [[SLICED:%[^ ]+]] = map.index.slice.stride<#map.index.slice.spec<[10, 64] -> [3, 64], {{.*}}>> ([[X]],
// CHECK-NEXT:      [[VIEW:%[^ ]+]] = map.index.reshape.stride<#map.index.reshape.spec<[3, 64] -> [192]>> ([[SLICED]])
// CHECK-NEXT:      map.call.opaque @kernels.relu[] inputs([[VIEW]]) outputs([[OUT]])
mo.graph @reshapeOfSlice(%x: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[192], f32> {
  %start = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %stop = mo.constant <#M.dense_array<4> : tensor<si64>> : !mo.tensor<[], si64>
  %s = mogg.slice(%x, %start, %stop) : (!mo.tensor<[10, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32>
  %r = mo.static.reshape(%s) : !mo.tensor<[3, 64], f32> -> !mo.tensor<[192], f32>
  %res = map.iter.opaque
      inputs (%r : !mo.tensor<[192], f32>)
      outputs (%out : !mo.tensor<[192], f32>) {
    map.call.opaque @kernels.relu[] inputs (%r) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[192], f32>) -> (!mo.tensor<[192], f32>)
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %res : !mo.tensor<[192], f32>
}

// A slice with bounds known only at run time stays.

// CHECK-LABEL: mo.graph @runtimeBounds
// CHECK:         [[S:%[^ ]+]] = mogg.slice
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[S]]: !mo.tensor<[3, 64], f32>)
// CHECK-NOT:       map.index.slice.stride
// CHECK:         mo.output
mo.graph @runtimeBounds(%x: !mo.tensor<[10, 64], f32>, %start: !mo.tensor<[], si64>, %stop: !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32> {
  %s = mogg.slice(%x, %start, %stop) : (!mo.tensor<[10, 64], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, 64], f32>
  %res = map.iter.opaque
      inputs (%s : !mo.tensor<[3, 64], f32>)
      outputs (%out : !mo.tensor<[3, 64], f32>) {
    map.call.opaque @kernels.relu[] inputs (%s) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 64], f32>) -> (!mo.tensor<[3, 64], f32>)
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %res : !mo.tensor<[3, 64], f32>
}

// So does a slice with parametric shapes.

// CHECK-LABEL: mo.graph @parametricShape
// CHECK:         [[S:%[^ ]+]] = mogg.slice
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[S]]: !mo.tensor<[3, D0], f32>)
// CHECK-NOT:       map.index.slice.stride
// CHECK:         mo.output
mo.graph @parametricShape<D0>(%x: !mo.tensor<[10, D0], f32>) -> !mo.tensor<[3, D0], f32> {
  %start = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %stop = mo.constant <#M.dense_array<4> : tensor<si64>> : !mo.tensor<[], si64>
  %s = mogg.slice(%x, %start, %stop) : (!mo.tensor<[10, D0], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[3, D0], f32>
  %res = map.iter.opaque
      inputs (%s : !mo.tensor<[3, D0], f32>)
      outputs (%out : !mo.tensor<[3, D0], f32>) {
    map.call.opaque @kernels.relu[] inputs (%s) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, D0], f32>) -> (!mo.tensor<[3, D0], f32>)
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %res : !mo.tensor<[3, D0], f32>
}
