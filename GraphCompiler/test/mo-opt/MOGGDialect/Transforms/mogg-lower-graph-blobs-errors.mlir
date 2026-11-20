// RUN: mo-opt %s -split-input-file -verify-diagnostics \
// RUN:   --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mogg-lower-graph-blobs

// A slice view reads its bounds on the host, so the kernel can't build it.
mo.graph @slice(%arg0: !mo.tensor<[10, 64], f32>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[3, 32], f32> {
  %0 = mogg.routine (%arg0, %arg1, %arg1, %arg1 : !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 32], f32>
    inputs (%a, %start, %stop, %step)
    outputs (%b)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s0 = mogg.shape.create() <shape = #mosh<ape[10, 64]> : !mosh.ape>
      %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[10, 64]> : !mosh.ape> : !mogg.tensor
      %s1 = mogg.shape.create() <shape = #mosh<ape[2]> : !mosh.ape>
      %st = mogg.tensor.create(%start, %s1) <device = "cpu", elementType = si64, io = Input, mut = false, shapeAttr = #mosh<ape[2]> : !mosh.ape> : !mogg.tensor
      %sp = mogg.tensor.create(%stop, %s1) <device = "cpu", elementType = si64, io = Input, mut = false, shapeAttr = #mosh<ape[2]> : !mosh.ape> : !mogg.tensor
      %se = mogg.tensor.create(%step, %s1) <device = "cpu", elementType = si64, io = Input, mut = false, shapeAttr = #mosh<ape[2]> : !mosh.ape> : !mogg.tensor
      // expected-note @below {{built here}}
      %a1 = mogg.tensor.create.slice(%a0, %st, %sp, %se) <elementType = f32, mut = false, outShape = #mosh<ape[3, 32]> : !mosh.ape, staticStart = array<i64: -9223372036854775808, 0>, staticStep = array<i64: 1, 1>> : !mogg.tensor
      %s2 = mogg.shape.create() <shape = #mosh<ape[3, 32]> : !mosh.ape>
      %b0 = mogg.tensor.create(%b, %s2) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[3, 32]> : !mosh.ape> : !mogg.tensor
      %a2 = mogg.trait.create(%a1) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      // expected-error @below {{reads a value that the host builds and a graph blob launch can't take}}
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a2 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0 : !mo.tensor<[3, 32], f32>
}

// -----

// A transpose view with a run-time permutation reads it on the host.
mo.graph @transpose(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32> {
  %0 = mogg.routine (%arg0, %arg1 : !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
    inputs (%a, %perm)
    outputs (%b)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s0 = mogg.shape.create() <shape = #mosh<ape[2, 3]> : !mosh.ape>
      %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 3]> : !mosh.ape> : !mogg.tensor
      %s1 = mogg.shape.create() <shape = #mosh<ape[2]> : !mosh.ape>
      %p = mogg.tensor.create(%perm, %s1) <device = "cpu", elementType = si64, io = Input, mut = false, shapeAttr = #mosh<ape[2]> : !mosh.ape> : !mogg.tensor
      // expected-note @below {{built here}}
      %a1 = mogg.tensor.create.transpose(%a0, %p) <elementType = f32, mut = false, outShape = #mosh<ape[3, 2]> : !mosh.ape, staticPermutations = array<i64: -9223372036854775808, -9223372036854775808>> : !mogg.tensor
      %s2 = mogg.shape.create() <shape = #mosh<ape[3, 2]> : !mosh.ape>
      %b0 = mogg.tensor.create(%b, %s2) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[3, 2]> : !mosh.ape> : !mogg.tensor
      %a2 = mogg.trait.create(%a1) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      // expected-error @below {{reads a value that the host builds and a graph blob launch can't take}}
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a2 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0 : !mo.tensor<[3, 2], f32>
}
