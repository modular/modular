// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mogg-form-graph-blobs | FileCheck %s

// A fused routine is deferred if any member is.
// CHECK-LABEL: mo.graph @one_deferred
// CHECK:       mogg.routine (
// CHECK:         mogg.graph_blob {
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative
// CHECK-NEXT:    }
// CHECK-NOT:   mogg.routine
// CHECK:       mo.output
mo.graph @one_deferred(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0 = mogg.routine dispatch(immediate) (%arg0 : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs (%a)
    outputs (%b)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %s1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %b0 = mogg.tensor.create(%b, %s1) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a0 : !mogg.tensor) outputs(%b0 : !mogg.tensor) context(%ctx : !mogg.context)
  }
  %1 = mogg.routine (%0 : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs (%a)
    outputs (%b)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %s1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %b0 = mogg.tensor.create(%b, %s1) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a0 : !mogg.tensor) outputs(%b0 : !mogg.tensor) context(%ctx : !mogg.context)
  }
  mo.output %1 : !mo.tensor<[2, 2], f32>
}

// A fused routine whose members are all immediate is immediate too.
// CHECK-LABEL: mo.graph @all_immediate
// CHECK:       mogg.routine dispatch(immediate) (
// CHECK:         mogg.graph_blob {
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative
// CHECK:           mogg.call.execute_device @{{.*}}MapLowerDeclarative
// CHECK-NEXT:    }
// CHECK-NOT:   mogg.routine
// CHECK:       mo.output
mo.graph @all_immediate(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0 = mogg.routine dispatch(immediate) (%arg0 : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs (%a)
    outputs (%b)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %s1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %b0 = mogg.tensor.create(%b, %s1) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a0 : !mogg.tensor) outputs(%b0 : !mogg.tensor) context(%ctx : !mogg.context)
  }
  %1 = mogg.routine dispatch(immediate) (%0 : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs (%a)
    outputs (%b)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %s1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %b0 = mogg.tensor.create(%b, %s1) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a0 : !mogg.tensor) outputs(%b0 : !mogg.tensor) context(%ctx : !mogg.context)
  }
  mo.output %1 : !mo.tensor<[2, 2], f32>
}
