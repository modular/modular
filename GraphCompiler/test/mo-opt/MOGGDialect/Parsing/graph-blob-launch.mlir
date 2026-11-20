// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   | mo-opt | FileCheck %s
//
// The calls name kernel declarations that only exist once
// `--mogg-import-packages` has run, so the first parse doesn't verify. The
// second parse reads the printed IR back with the verifier on.

// A lowered graph blob: the host computes each call's configuration, the
// launch configuration and each call's active count, then launches one kernel.
// The kernel takes the routine's pointers and the active counts, and each call
// builds its own operands inside its call region.

// CHECK-LABEL: mo.graph @blob
// CHECK:      %[[CFG0:.+]] = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%{{.+}} : !mogg.tensor) outputs(%{{.+}} : !mogg.tensor)
// CHECK:      %[[CFG1:.+]] = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%{{.+}} : !mogg.tensor) outputs(%{{.+}} : !mogg.tensor)
// CHECK:      %[[LAUNCH:.+]] = mogg.graph_blob.launch_config context(%[[CTX:[^)]+]])
// CHECK-NEXT: %[[ACT0:.+]] = mogg.graph_blob.active_threads %[[CFG0]] in %[[LAUNCH]]
// CHECK-NEXT: %[[ACT1:.+]] = mogg.graph_blob.active_threads %[[CFG1]] in %[[LAUNCH]]
// CHECK-NEXT: mogg.graph_blob.launch context(%[[CTX]]) config(%[[LAUNCH]]) args {%[[X:[^ ]+]] = %{{[^ ]+}} : !mogg.pointer, %[[Y:[^ ]+]] = %{{[^ ]+}} : !mogg.pointer, %[[Z:[^ ]+]] = %{{[^ ]+}} : !mogg.pointer, %[[A0:[^ ]+]] = %[[ACT0]] : i32, %[[A1:[^ ]+]] = %[[ACT1]] : i32} {
// CHECK-NEXT:   mogg.graph_blob.call active(%[[A0]]) {
// CHECK:          mogg.tensor.create(%[[X]],
// CHECK:          mogg.tensor.create(%[[Y]],
// CHECK:          mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%{{.+}} : !mogg.tensor) outputs(%{{.+}} : !mogg.tensor){{$}}
// CHECK-NEXT:   }
// CHECK-NEXT:   mogg.graph_blob.barrier
// CHECK-NEXT:   mogg.graph_blob.call active(%[[A1]]) {
// CHECK:          mogg.tensor.create(%[[Y]],
// CHECK:          mogg.tensor.create(%[[Z]],
// CHECK:          mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%{{.+}} : !mogg.tensor) outputs(%{{.+}} : !mogg.tensor){{$}}
// CHECK-NEXT:   }
// CHECK-NEXT: }
mo.graph @blob(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    %s = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %a_in = mogg.tensor.create(%a, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %b_out = mogg.tensor.create(%b, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %cfg0 = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a_in : !mogg.tensor) outputs(%b_out : !mogg.tensor)
    %b_in = mogg.tensor.create(%b, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %c_out = mogg.tensor.create(%c, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %cfg1 = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b_in : !mogg.tensor) outputs(%c_out : !mogg.tensor)
    %launch = mogg.graph_blob.launch_config context(%ctx)
    %act0 = mogg.graph_blob.active_threads %cfg0 in %launch
    %act1 = mogg.graph_blob.active_threads %cfg1 in %launch
    mogg.graph_blob.launch context(%ctx) config(%launch)
        args {%x = %a : !mogg.pointer, %y = %b : !mogg.pointer, %z = %c : !mogg.pointer, %a0 = %act0 : i32, %a1 = %act1 : i32} {
      mogg.graph_blob.call active(%a0) {
        %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
        %t0 = mogg.tensor.create(%x, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
        %t1 = mogg.tensor.create(%y, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
        %in = mogg.trait.create(%t0) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
        %out = mogg.trait.create(%t1) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
        mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor)
      }
      mogg.graph_blob.barrier
      mogg.graph_blob.call active(%a1) {
        %s1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
        %t2 = mogg.tensor.create(%y, %s1) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
        %t3 = mogg.tensor.create(%z, %s1) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
        %in = mogg.trait.create(%t2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
        %out = mogg.trait.create(%t3) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
        mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor)
      }
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32>
}

// A dim crosses the launch like any other operand.

// CHECK-LABEL: mo.graph @blob_dim
// CHECK:      mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%[[X:[^ ]+]] = %{{[^ ]+}} : !mogg.pointer, %[[Y:[^ ]+]] = %{{[^ ]+}} : !mogg.pointer, %[[N:[^ ]+]] = %{{[^ ]+}} : !mogg.dim<batch>, %[[A:[^ ]+]] = %{{[^ ]+}} : i32} {
// CHECK-NEXT:   mogg.graph_blob.call active(%[[A]]) {
// CHECK-NEXT:     mogg.shape.create(%[[N]])
mo.graph @blob_dim<batch>(%arg0: !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32> {
  %0 = mogg.routine (%arg0 : !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32>
    inputs (%a)
    outputs (%b)
    context (%ctx)
    dims (%n : !mogg.dim<batch>) attributes {device = #M.device_ref<"gpu", 0>} {
    %s = mogg.shape.create(%n) <shape = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.dim<batch>
    %a_in = mogg.tensor.create(%a, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
    %b_out = mogg.tensor.create(%b, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
    %cfg = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a_in : !mogg.tensor) outputs(%b_out : !mogg.tensor)
    %launch = mogg.graph_blob.launch_config context(%ctx)
    %act = mogg.graph_blob.active_threads %cfg in %launch
    mogg.graph_blob.launch context(%ctx) config(%launch)
        args {%x = %a : !mogg.pointer, %y = %b : !mogg.pointer, %d = %n : !mogg.dim<batch>, %a0 = %act : i32} {
      mogg.graph_blob.call active(%a0) {
        %s0 = mogg.shape.create(%d) <shape = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.dim<batch>
        %t0 = mogg.tensor.create(%x, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
        %t1 = mogg.tensor.create(%y, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
        %in = mogg.trait.create(%t0) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
        %out = mogg.trait.create(%t1) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
        mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%in : !mogg.tensor) outputs(%out : !mogg.tensor)
      }
    }
  }
  mo.output %0 : !mo.tensor<[batch, 2], f32>
}
