// RUN: mo-opt %s --pass-pipeline='builtin.module(mo.graph(map-fuse-contiguous-views, canonicalize))' | FileCheck %s

// FuseContiguousViews rewrites a kernel input that a mo.static.reshape
// produces to read the reshape's input through a map.index.reshape.stride at
// the top of the kernel's body. This covers a map.call.opaque with no
// prologue, which takes its operands whole. The canonicalizer then drops the
// old input and the reshape once nothing reads them, as in the pipeline.

// map.call.opaque verifies that its callee resolves to a mogg.kernel.decl. The
// pass never reads the signature, so empty declarations are enough here.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" struct_params = [] is_user_defined = false
mogg.kernel.decl @kernels.add registration = "kernels.add" struct_params = [] is_user_defined = false

// CHECK-LABEL: mo.graph @reshapeIntoCall
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK-NOT:     mo.static.reshape
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[2, 6], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 4], f32>) {
// CHECK-NEXT:      [[VIEW:%[^ ]+]] = map.index.reshape.stride<#map.index.reshape.spec<[2, 6] -> [3, 4]>> ([[X]]) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[3, 4], f32>
// CHECK-NEXT:      map.call.opaque @kernels.relu[] inputs([[VIEW]]) outputs([[OUT]])
mo.graph @reshapeIntoCall(%x: !mo.tensor<[2, 6], f32>) -> !mo.tensor<[3, 4], f32> {
  %r = mo.static.reshape(%x) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[3, 4], f32>
  %res = map.iter.opaque
      inputs (%r : !mo.tensor<[3, 4], f32>)
      outputs (%out : !mo.tensor<[3, 4], f32>) {
    map.call.opaque @kernels.relu[] inputs (%r) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 4], f32>) -> (!mo.tensor<[3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[3, 4], f32>
}

// The reshaped input sits among untouched ones; the source is appended and the
// other inputs keep their order.

// CHECK-LABEL: mo.graph @reshapeAmongInputs
// CHECK-SAME:  ([[A:%[^:]+]]: {{.*}}, [[X:%[^:]+]]:
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[A]]: !mo.tensor<[3, 4], f32>, [[X]]: !mo.tensor<[12], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 4], f32>) {
// CHECK-NEXT:      [[VIEW:%[^ ]+]] = map.index.reshape.stride<#map.index.reshape.spec<[12] -> [3, 4]>> ([[X]])
// CHECK-NEXT:      map.call.opaque @kernels.add[] inputs([[A]], [[VIEW]]) outputs([[OUT]])
mo.graph @reshapeAmongInputs(%a: !mo.tensor<[3, 4], f32>, %x: !mo.tensor<[12], f32>) -> !mo.tensor<[3, 4], f32> {
  %r = mo.static.reshape(%x) : !mo.tensor<[12], f32> -> !mo.tensor<[3, 4], f32>
  %res = map.iter.opaque
      inputs (%a : !mo.tensor<[3, 4], f32>, %r : !mo.tensor<[3, 4], f32>)
      outputs (%out : !mo.tensor<[3, 4], f32>) {
    map.call.opaque @kernels.add[] inputs (%a, %r) outputs (%out) kernelName = "kernels.add" : (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) -> (!mo.tensor<[3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[3, 4], f32>
}

// Each kernel that reads the reshape gets its own view, and the reshape goes
// once neither reads it.

// CHECK-LABEL: mo.graph @twoKernels
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK-NOT:     mo.static.reshape
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[2, 6], f32>)
// CHECK:           map.index.reshape.stride<#map.index.reshape.spec<[2, 6] -> [12]>> ([[X]])
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[2, 6], f32>)
// CHECK:           map.index.reshape.stride<#map.index.reshape.spec<[2, 6] -> [12]>> ([[X]])
mo.graph @twoKernels(%x: !mo.tensor<[2, 6], f32>) -> (!mo.tensor<[12], f32>, !mo.tensor<[12], f32>) {
  %r = mo.static.reshape(%x) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[12], f32>
  %res0 = map.iter.opaque
      inputs (%r : !mo.tensor<[12], f32>)
      outputs (%out : !mo.tensor<[12], f32>) {
    map.call.opaque @kernels.relu[] inputs (%r) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[12], f32>) -> (!mo.tensor<[12], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  %res1 = map.iter.opaque
      inputs (%r : !mo.tensor<[12], f32>)
      outputs (%out : !mo.tensor<[12], f32>) {
    map.call.opaque @kernels.relu[] inputs (%r) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[12], f32>) -> (!mo.tensor<[12], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res0, %res1 : !mo.tensor<[12], f32>, !mo.tensor<[12], f32>
}

// A kernel that reads the same reshape twice takes the source once.

// CHECK-LABEL: mo.graph @sameReshapeTwice
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[12], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[3, 4], f32>) {
// CHECK-NEXT:      [[VIEW:%[^ ]+]] = map.index.reshape.stride
// CHECK-NEXT:      map.call.opaque @kernels.add[] inputs([[VIEW]], [[VIEW]]) outputs([[OUT]])
mo.graph @sameReshapeTwice(%x: !mo.tensor<[12], f32>) -> !mo.tensor<[3, 4], f32> {
  %r = mo.static.reshape(%x) : !mo.tensor<[12], f32> -> !mo.tensor<[3, 4], f32>
  %res = map.iter.opaque
      inputs (%r as %r0 : !mo.tensor<[3, 4], f32>, %r as %r1 : !mo.tensor<[3, 4], f32>)
      outputs (%out : !mo.tensor<[3, 4], f32>) {
    map.call.opaque @kernels.add[] inputs (%r0, %r1) outputs (%out) kernelName = "kernels.add" : (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) -> (!mo.tensor<[3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[3, 4], f32>
}

// The graph's output still reads the reshape, so it stays for that reader.

// CHECK-LABEL: mo.graph @keptForOutput
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK:         [[R:%[^ ]+]] = mo.static.reshape([[X]])
// CHECK:         [[RES:%[^ ]+]] = map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[2, 6], f32>)
// CHECK:           map.index.reshape.stride<#map.index.reshape.spec<[2, 6] -> [3, 4]>> ([[X]])
// CHECK:         mo.output [[RES]], [[R]]
mo.graph @keptForOutput(%x: !mo.tensor<[2, 6], f32>) -> (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) {
  %r = mo.static.reshape(%x) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[3, 4], f32>
  %res = map.iter.opaque
      inputs (%r : !mo.tensor<[3, 4], f32>)
      outputs (%out : !mo.tensor<[3, 4], f32>) {
    map.call.opaque @kernels.relu[] inputs (%r) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 4], f32>) -> (!mo.tensor<[3, 4], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res, %r : !mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>
}

// A structured kernel reads the input through a closure capture, which takes
// the view in its place.

// CHECK-LABEL: mo.graph @reshapeIntoCapture
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK:         map.iter.opaque
// CHECK-NEXT:    inputs ([[X]]: !mo.tensor<[2, 6], f32>)
// CHECK-NEXT:    outputs ([[OUT:%[^ :,]+]] : !mo.tensor<[12], f32>) {
// CHECK-NEXT:      [[VIEW:%[^ ]+]] = map.index.reshape.stride<#map.index.reshape.spec<[2, 6] -> [12]>> ([[X]])
// CHECK-NEXT:      map.iter.elementwise.body<[12]>
// CHECK-NEXT:        index
// CHECK-NEXT:        capturing {{{.*}} = [[VIEW]] : !mo.tensor<[12], f32>, {{.*}} = [[OUT]] : !mo.tensor<[12], f32>}
mo.graph @reshapeIntoCapture(%x: !mo.tensor<[2, 6], f32>) -> !mo.tensor<[12], f32> {
  %r = mo.static.reshape(%x) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[12], f32>
  %res = map.iter.opaque
      inputs (%r : !mo.tensor<[12], f32>)
      outputs (%out : !mo.tensor<[12], f32>) {
    map.iter.elementwise.body<[12]>
        index (%idx: !map.index<1>)
        capturing {%r0 = %r : !mo.tensor<[12], f32>, %out0 = %out : !mo.tensor<[12], f32>} {
      %val = map.load %r0[%idx] : !mo.tensor<[12], f32>
      map.store %out0[%idx] = %val : !mo.tensor<[12], f32>
    }
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[12], f32>
}

// The view carries the reshape's parametric shapes.

// CHECK-LABEL: mo.graph @dynamicReshape<D0>
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK-NOT:     mo.static.reshape
// CHECK:           map.index.reshape.stride<#map.index.reshape.spec<[D0, 4] -> [mul_no_wrap(D0, 2), 2]>> ([[X]]) : !mo.tensor<[D0, 4], f32> -> !mo.tensor<[mul_no_wrap(D0, 2), 2], f32>
mo.graph @dynamicReshape<D0>(%x: !mo.tensor<[D0, 4], f32>) -> !mo.tensor<[mul_no_wrap(D0, 2), 2], f32> {
  %r = mo.static.reshape(%x) : !mo.tensor<[D0, 4], f32> -> !mo.tensor<[mul_no_wrap(D0, 2), 2], f32>
  %res = map.iter.opaque
      inputs (%r : !mo.tensor<[mul_no_wrap(D0, 2), 2], f32>)
      outputs (%out : !mo.tensor<[mul_no_wrap(D0, 2), 2], f32>) {
    map.call.opaque @kernels.relu[] inputs (%r) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[mul_no_wrap(D0, 2), 2], f32>) -> (!mo.tensor<[mul_no_wrap(D0, 2), 2], f32>)
  } {device = #M.device_ref<"cpu", 0>}
  mo.output %res : !mo.tensor<[mul_no_wrap(D0, 2), 2], f32>
}

// A kernel nested in a region fuses a reshape defined outside the region as
// well as one beside it.

// CHECK-LABEL: mo.graph @otherBlock
// CHECK-SAME:  ([[X:%[^:]+]]:
// CHECK-NOT:     mo.static.reshape
// CHECK:         mo.if
// CHECK:           map.iter.opaque
// CHECK-NEXT:      inputs ([[X]]: !mo.tensor<[2, 6], f32>)
// CHECK-NEXT:      outputs
// CHECK-NEXT:        map.index.reshape.stride<#map.index.reshape.spec<[2, 6] -> [3, 4]>> ([[X]])
// CHECK:         } else {
// CHECK:           map.iter.opaque
// CHECK-NEXT:      inputs ([[X]]: !mo.tensor<[2, 6], f32>)
// CHECK-NEXT:      outputs
// CHECK-NEXT:        map.index.reshape.stride<#map.index.reshape.spec<[2, 6] -> [3, 4]>> ([[X]])
mo.graph @otherBlock(%x: !mo.tensor<[2, 6], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[3, 4], f32> {
  %r = mo.static.reshape(%x) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[3, 4], f32>
  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[3, 4], f32> {
    %res = map.iter.opaque
        inputs (%r : !mo.tensor<[3, 4], f32>)
        outputs (%out : !mo.tensor<[3, 4], f32>) {
      map.call.opaque @kernels.relu[] inputs (%r) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 4], f32>) -> (!mo.tensor<[3, 4], f32>)
    } {device = #M.device_ref<"cpu", 0>}
    mo.yield %res : !mo.tensor<[3, 4], f32>
  } else {
    %r2 = mo.static.reshape(%x) : !mo.tensor<[2, 6], f32> -> !mo.tensor<[3, 4], f32>
    %res = map.iter.opaque
        inputs (%r2 : !mo.tensor<[3, 4], f32>)
        outputs (%out : !mo.tensor<[3, 4], f32>) {
      map.call.opaque @kernels.relu[] inputs (%r2) outputs (%out) kernelName = "kernels.relu" : (!mo.tensor<[3, 4], f32>) -> (!mo.tensor<[3, 4], f32>)
    } {device = #M.device_ref<"cpu", 0>}
    mo.yield %res : !mo.tensor<[3, 4], f32>
  }
  mo.output %0 : !mo.tensor<[3, 4], f32>
}
