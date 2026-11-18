// RUN: mo-opt %s --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mogg-lower-graph-blobs --canonicalize | FileCheck %s
//
// The calls name kernel declarations that only exist once
// `--mogg-import-packages` has run, so parse-time verification is disabled.

// Two calls, the second reading the first's output. The host computes both
// configurations and active counts, then launches one kernel. The kernel takes
// each pointer once, and each call region rebuilds its own operands.

// CHECK-LABEL: mo.graph @sequence
// CHECK:        inputs (%[[A:[^ ,)]+]])
// CHECK-NEXT:   outputs (%[[B:[^ ,)]+]], %[[C:[^ ,)]+]])
// CHECK-NEXT:   context (%[[CTX:[^ ,)]+]])
// CHECK-NOT:    mogg.graph_blob {
// CHECK:        %[[CFG0:.+]] = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%{{.+}} : !mogg.tensor) outputs(%{{.+}} : !mogg.tensor){{$}}
// CHECK-NEXT:   %[[CFG1:.+]] = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%{{.+}} : !mogg.tensor) outputs(%{{.+}} : !mogg.tensor){{$}}
// CHECK-NEXT:   %[[LAUNCH:.+]] = mogg.graph_blob.launch_config context(%[[CTX]])
// CHECK-NEXT:   %[[ACT0:.+]] = mogg.graph_blob.active_threads %[[CFG0]] in %[[LAUNCH]]
// CHECK-NEXT:   %[[ACT1:.+]] = mogg.graph_blob.active_threads %[[CFG1]] in %[[LAUNCH]]
// CHECK-NEXT:   mogg.graph_blob.launch context(%[[CTX]]) config(%[[LAUNCH]]) args {%[[X:[^ ]+]] = %[[A]] : !mogg.pointer, %[[Y:[^ ]+]] = %[[B]] : !mogg.pointer, %[[Z:[^ ]+]] = %[[C]] : !mogg.pointer, %[[A0:[^ ]+]] = %[[ACT0]] : i32, %[[A1:[^ ]+]] = %[[ACT1]] : i32} {
// CHECK-NEXT:     mogg.graph_blob.call active(%[[A0]]) {
// CHECK-NEXT:       %[[S0:.+]] = mogg.shape.create()
// CHECK-NEXT:       %[[T0:.+]] = mogg.tensor.create(%[[X]], %[[S0]])
// CHECK-NEXT:       %[[IN0:.+]] = mogg.trait.create(%[[T0]])
// CHECK-NEXT:       %[[S1:.+]] = mogg.shape.create()
// CHECK-NEXT:       %[[T1:.+]] = mogg.tensor.create(%[[Y]], %[[S1]])
// CHECK-NEXT:       %[[OUT0:.+]] = mogg.trait.create(%[[T1]])
// CHECK-NEXT:       mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%[[IN0]] : !mogg.tensor) outputs(%[[OUT0]] : !mogg.tensor){{$}}
// CHECK-NEXT:     }
// CHECK-NEXT:     mogg.graph_blob.barrier
// CHECK-NEXT:     mogg.graph_blob.call active(%[[A1]]) {
// CHECK-NEXT:       %[[S2:.+]] = mogg.shape.create()
// CHECK-NEXT:       %[[T2:.+]] = mogg.tensor.create(%[[Y]], %[[S2]])
// CHECK-NEXT:       %[[IN1:.+]] = mogg.trait.create(%[[T2]])
// CHECK-NEXT:       %[[S3:.+]] = mogg.shape.create()
// CHECK-NEXT:       %[[T3:.+]] = mogg.tensor.create(%[[Z]], %[[S3]])
// CHECK-NEXT:       %[[OUT1:.+]] = mogg.trait.create(%[[T3]])
// CHECK-NEXT:       mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%[[IN1]] : !mogg.tensor) outputs(%[[OUT1]] : !mogg.tensor){{$}}
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
mo.graph @sequence(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %s1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %b0 = mogg.tensor.create(%b, %s1) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %a1 = mogg.trait.create(%a0) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a1 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
      %s2 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %b2 = mogg.tensor.create(%b, %s2) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %s3 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %c0 = mogg.tensor.create(%c, %s3) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b3 = mogg.trait.create(%b2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %c1 = mogg.trait.create(%c0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b3 : !mogg.tensor) outputs(%c1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32>
}

// A named dim that both calls' shapes use crosses once, with its name.

// CHECK-LABEL: mo.graph @dim
// CHECK:        dims (%[[N:[^ :]+]]: !mogg.dim<batch>)
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%{{[^ ]+}} = %{{[^ ]+}} : !mogg.pointer, %[[D:[^ ]+]] = %[[N]] : !mogg.dim<batch>, %{{[^ ]+}} = %{{[^ ]+}} : !mogg.pointer, %{{[^ ]+}} = %{{[^ ]+}} : !mogg.pointer, %{{[^ ]+}} = %{{[^ ]+}} : i32, %{{[^ ]+}} = %{{[^ ]+}} : i32} {
// CHECK-NEXT:     mogg.graph_blob.call
// CHECK:            mogg.shape.create(%[[D]])
// CHECK:          mogg.graph_blob.barrier
// CHECK-NEXT:     mogg.graph_blob.call
// CHECK:            mogg.shape.create(%[[D]])
mo.graph @dim<batch>(%arg0: !mo.tensor<[batch, 2], f32, gpu:0>) -> !mo.tensor<[batch, 2], f32, gpu:0> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[batch, 2], f32, gpu:0>) -> (!mo.tensor<[batch, 2], f32, gpu:0>, !mo.tensor<[batch, 2], f32, gpu:0>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx)
    dims (%n: !mogg.dim<batch>) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s0 = mogg.shape.create(%n) <shape = #mosh<ape[dyn, 2]> : !mosh.ape> : !mogg.dim<batch>
      %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
      %s1 = mogg.shape.create(%n) <shape = #mosh<ape[dyn, 2]> : !mosh.ape> : !mogg.dim<batch>
      %b0 = mogg.tensor.create(%b, %s1) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
      %a1 = mogg.trait.create(%a0) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a1 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
      %s2 = mogg.shape.create(%n) <shape = #mosh<ape[dyn, 2]> : !mosh.ape> : !mogg.dim<batch>
      %b2 = mogg.tensor.create(%b, %s2) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
      %s3 = mogg.shape.create(%n) <shape = #mosh<ape[dyn, 2]> : !mosh.ape> : !mogg.dim<batch>
      %c0 = mogg.tensor.create(%c, %s3) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
      %b3 = mogg.trait.create(%b2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %c1 = mogg.trait.create(%c0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b3 : !mogg.tensor) outputs(%c1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[batch, 2], f32, gpu:0>
}

// The host reads a scalar operand's element, and the scalar crosses by value.
// The tensor it's read from doesn't cross.

// CHECK-LABEL: mo.graph @scalar
// CHECK:        inputs (%[[A:[^ ,)]+]], %{{[^ ,)]+}})
// CHECK-NEXT:   outputs (%[[B:[^ ,)]+]], %[[C:[^ ,)]+]])
// CHECK:        %[[SC:.+]] = mogg.scalar.create(
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%{{[^ ]+}} = %[[A]] : !mogg.pointer, %{{[^ ]+}} = %[[B]] : !mogg.pointer, %[[SCARG:[^ ]+]] = %[[SC]] : !mogg.scalar, %{{[^ ]+}} = %[[C]] : !mogg.pointer, %{{[^ ]+}} = %{{[^ ]+}} : i32, %{{[^ ]+}} = %{{[^ ]+}} : i32} {
// CHECK-NOT:      mogg.scalar.create
// CHECK:          mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarativeScalar inputs(%{{[^,]+}}, %[[SCARG]] : !mogg.tensor, !mogg.scalar)
mo.graph @scalar(%arg0: !mo.tensor<[2, 2], f32, gpu:0>, %arg1: !mo.tensor<[], f32>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %0:2 = mogg.routine (%arg0, %arg1 : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[], f32>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>)
    inputs (%a, %s)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %2 = mogg.tensor.create(%a, %1) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %3 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %4 = mogg.tensor.create(%b, %3) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %5 = mogg.trait.create(%2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %6 = mogg.trait.create(%4) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%5 : !mogg.tensor) outputs(%6 : !mogg.tensor) context(%ctx : !mogg.context)
      %7 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %8 = mogg.tensor.create(%b, %7) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %9 = mogg.shape.create() <shape = #mosh<ape[]> : !mosh.ape>
      %10 = mogg.tensor.create(%s, %9) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[]> : !mosh.ape> : !mogg.tensor
      %11 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %12 = mogg.tensor.create(%c, %11) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %13 = mogg.trait.create(%8) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %14 = mogg.scalar.create(%10)
      %15 = mogg.trait.create(%12) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarativeScalar inputs(%13, %14 : !mogg.tensor, !mogg.scalar) outputs(%15 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32, gpu:0>
}

// A fused output stays on the host and crosses as itself. The tensors its
// epilogue captures don't cross on their own: the bias never does, and the
// first call's output crosses only because the second call reads it.

// CHECK-LABEL: mo.graph @fused
// CHECK:        inputs (%[[A:[^ ,)]+]], %{{[^ ,)]+}})
// CHECK-NEXT:   outputs (%[[C:[^ ,)]+]], %{{[^ ,)]+}})
// CHECK:        %[[EPI0:.+]] = mogg.tensor.create.epilogue
// CHECK:        %[[EPI1:.+]] = mogg.tensor.create.epilogue
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%{{[^ ]+}} = %[[A]] : !mogg.pointer, %[[O0:[^ ]+]] = %[[EPI0]] : !mogg.tensor, %[[Y:[^ ]+]] = %[[C]] : !mogg.pointer, %[[O1:[^ ]+]] = %[[EPI1]] : !mogg.tensor, %{{[^ ]+}} = %{{[^ ]+}} : i32, %{{[^ ]+}} = %{{[^ ]+}} : i32} {
// CHECK-NOT:      mogg.tensor.create.epilogue
// CHECK:          mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarativeFusedStore inputs(%{{.+}} : !mogg.tensor) outputs(%[[O0]] : !mogg.tensor){{$}}
// CHECK:          mogg.graph_blob.barrier
// CHECK:          mogg.tensor.create(%[[Y]],
// CHECK:          mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarativeFusedStore inputs(%{{.+}} : !mogg.tensor) outputs(%[[O1]] : !mogg.tensor){{$}}
mo.graph @fused(%arg0: !mo.tensor<[2, 2], f32, gpu:0>, %arg1: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %0:2 = mogg.routine (%arg0, %arg1 : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>)
    inputs (%a, %bias)
    outputs (%c, %e)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %2 = mogg.tensor.create(%a, %1) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %3 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %4 = mogg.tensor.create(%bias, %3) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %5 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %6 = mogg.tensor.create(%c, %5) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %7 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %8 = mogg.tensor.get_ptr(%6) : !mogg.tensor
      %9 = mogg.tensor.create.epilogue %7 (%val : !mogg.simd<f32>, %idx : !map.index<2>) capturing {%cap0 = %6 : !mogg.tensor, %cap1 = %4 : !mogg.tensor} ptr %8 contract = <{role = output, fused = true, access = store, capabilities = store}> device = "gpu", elementType = f32, io = Output, mut = true {
        %21 = mogg.tensor.load %cap1[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
        %22 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%val, %21) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
        mogg.tensor.store %cap0[%idx] = %22 : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
      }
      %10 = mogg.trait.create(%2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarativeFusedStore inputs(%10 : !mogg.tensor) outputs(%9 : !mogg.tensor) context(%ctx : !mogg.context)
      %11 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %12 = mogg.tensor.create(%c, %11) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %13 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %14 = mogg.tensor.create(%bias, %13) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %15 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %16 = mogg.tensor.create(%e, %15) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %17 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %18 = mogg.tensor.get_ptr(%16) : !mogg.tensor
      %19 = mogg.tensor.create.epilogue %17 (%val : !mogg.simd<f32>, %idx : !map.index<2>) capturing {%cap0 = %16 : !mogg.tensor, %cap1 = %14 : !mogg.tensor} ptr %18 contract = <{role = output, fused = true, access = store, capabilities = store}> device = "gpu", elementType = f32, io = Output, mut = true {
        %21 = mogg.tensor.load %cap1[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
        %22 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%val, %21) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
        mogg.tensor.store %cap0[%idx] = %22 : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
      }
      %20 = mogg.trait.create(%12) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarativeFusedStore inputs(%20 : !mogg.tensor) outputs(%19 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32, gpu:0>
}

// A reshape view is rebuilt in the call region from the tensor it views.

// CHECK-LABEL: mo.graph @reshape
// CHECK:        inputs (%[[A:[^ ,)]+]])
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%[[X:[^ ]+]] = %[[A]] : !mogg.pointer,
// CHECK-NEXT:     mogg.graph_blob.call
// CHECK:            %[[T:.+]] = mogg.tensor.create(%[[X]],
// CHECK:            %[[R:.+]] = mogg.tensor.create.reshape(%[[T]],
// CHECK:            %[[IN:.+]] = mogg.trait.create(%[[R]])
// CHECK:            mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%[[IN]] : !mogg.tensor)
// CHECK:          mogg.graph_blob.barrier
mo.graph @reshape(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s4 = mogg.shape.create() <shape = #mosh<ape[4]> : !mosh.ape>
      %a0 = mogg.tensor.create(%a, %s4) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[4]> : !mosh.ape> : !mogg.tensor
      %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %a1 = mogg.tensor.create.reshape(%a0, %s0) <elementType = f32, mut = false, staticOutShape = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b0 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %a2 = mogg.trait.create(%a1) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a2 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
      %b2 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %c0 = mogg.tensor.create(%c, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b3 = mogg.trait.create(%b2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %c1 = mogg.trait.create(%c0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b3 : !mogg.tensor) outputs(%c1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32>
}

// A broadcast view is rebuilt in the call region from the tensor it views.

// CHECK-LABEL: mo.graph @broadcast
// CHECK:        inputs (%[[A:[^ ,)]+]])
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%[[X:[^ ]+]] = %[[A]] : !mogg.pointer,
// CHECK-NEXT:     mogg.graph_blob.call
// CHECK:            %[[T:.+]] = mogg.tensor.create(%[[X]],
// CHECK:            %[[V:.+]] = mogg.tensor.create.broadcast(%[[T]],
// CHECK:            %[[IN:.+]] = mogg.trait.create(%[[V]])
// CHECK:            mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%[[IN]] : !mogg.tensor)
// CHECK:          mogg.graph_blob.barrier
mo.graph @broadcast(%arg0: !mo.tensor<[1, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[1, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %s1 = mogg.shape.create() <shape = #mosh<ape[1, 2]> : !mosh.ape>
      %a0 = mogg.tensor.create(%a, %s1) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[1, 2]> : !mosh.ape> : !mogg.tensor
      %a1 = mogg.tensor.create.broadcast(%a0, %s0) <elementType = f32, mut = false, staticOutShape = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b0 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %a2 = mogg.trait.create(%a1) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a2 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
      %b2 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %c0 = mogg.tensor.create(%c, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b3 = mogg.trait.create(%b2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %c1 = mogg.trait.create(%c0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b3 : !mogg.tensor) outputs(%c1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32>
}

// A constant buffer stays on the host, and its pointer crosses.

// CHECK-LABEL: mo.graph @constant
// CHECK:        %[[K:.+]] = mogg.buffer.constant
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%[[KARG:[^ ]+]] = %[[K]] : !mogg.pointer,
// CHECK-NOT:      mogg.buffer.constant
// CHECK:          mogg.tensor.create(%[[KARG]],
// CHECK:          mogg.graph_blob.barrier
mo.graph @constant(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %k = mogg.buffer.constant dense<1.000000e+00> : tensor<2x2xf32>
      %a1 = mogg.tensor.create(%k, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b0 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %a2 = mogg.trait.create(%a1) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a2 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
      %b2 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %c0 = mogg.tensor.create(%c, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b3 = mogg.trait.create(%b2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %c1 = mogg.trait.create(%c0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b3 : !mogg.tensor) outputs(%c1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32>
}

// A view built over another tensor's pointer is rebuilt in the call region,
// pointer read included.

// CHECK-LABEL: mo.graph @get_ptr
// CHECK:        inputs (%[[A:[^ ,)]+]])
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%[[X:[^ ]+]] = %[[A]] : !mogg.pointer,
// CHECK-NEXT:     mogg.graph_blob.call
// CHECK:            %[[T:.+]] = mogg.tensor.create(%[[X]],
// CHECK:            %[[P:.+]] = mogg.tensor.get_ptr(%[[T]])
// CHECK:            mogg.tensor.create(%[[P]],
// CHECK:          mogg.graph_blob.barrier
mo.graph @get_ptr(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %p = mogg.tensor.get_ptr(%a0) : !mogg.tensor
      %a1 = mogg.tensor.create(%p, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b0 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %a2 = mogg.trait.create(%a1) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a2 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
      %b2 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %c0 = mogg.tensor.create(%c, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b3 = mogg.trait.create(%b2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %c1 = mogg.trait.create(%c0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b3 : !mogg.tensor) outputs(%c1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32>
}

// A fused input stays on the host and crosses as itself, like a fused output.
// The tensors its prologue captures don't cross.

// CHECK-LABEL: mo.graph @prologue
// CHECK:        outputs (%[[B:[^ ,)]+]], %[[C:[^ ,)]+]])
// CHECK:        %[[PRO:.+]] = mogg.tensor.create.prologue
// CHECK:        mogg.graph_blob.launch context(%{{[^)]+}}) config(%{{[^)]+}}) args {%[[PARG:[^ ]+]] = %[[PRO]] : !mogg.tensor, %{{[^ ]+}} = %[[B]] : !mogg.pointer, %{{[^ ]+}} = %[[C]] : !mogg.pointer, %{{[^ ]+}} = %{{[^ ]+}} : i32, %{{[^ ]+}} = %{{[^ ]+}} : i32} {
// CHECK-NOT:      mogg.tensor.create.prologue
// CHECK:          mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarativeFusedLoad inputs(%[[PARG]] : !mogg.tensor)
mo.graph @prologue(%arg0: !mo.tensor<[2, 2], f32, gpu:0>, %arg1: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %0:2 = mogg.routine (%arg0, %arg1 : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>)
    inputs (%a, %bias)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    mogg.graph_blob {
      %1 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %2 = mogg.tensor.create(%a, %1) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %3 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %4 = mogg.tensor.create(%bias, %3) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %5 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %6 = mogg.tensor.create(%b, %5) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %7 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %8 = mogg.tensor.create.prologue %7 (%idx : !map.index<2>) capturing {%cap0 = %2 : !mogg.tensor, %cap1 = %4 : !mogg.tensor} contract = <{role = input, fused = true, access = load, capabilities = load}> device = "gpu", elementType = f32, io = Input, mut = false {
        %16 = mogg.tensor.load %cap0[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
        %17 = mogg.tensor.load %cap1[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
        %18 = mogg.call.elementwise @builtin_kernels.elementwise.Add(%16, %17) moOpName = "mo.add" : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
        mogg.yield %18 : !mogg.simd<f32>
      }
      %9 = mogg.trait.create(%6) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarativeFusedLoad inputs(%8 : !mogg.tensor) outputs(%9 : !mogg.tensor) context(%ctx : !mogg.context)
      %10 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %11 = mogg.tensor.create(%b, %10) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %12 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %13 = mogg.tensor.create(%c, %12) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %14 = mogg.trait.create(%11) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %15 = mogg.trait.create(%13) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%14 : !mogg.tensor) outputs(%15 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32, gpu:0>
}

// A routine without a graph blob doesn't change.

// CHECK-LABEL: mo.graph @no_blob
// CHECK-NOT:    mogg.graph_blob
// CHECK:        mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%{{.+}} : !mogg.tensor) outputs(%{{.+}} : !mogg.tensor) context(%{{.+}} : !mogg.context)
// CHECK-NOT:    mogg.graph_blob
mo.graph @no_blob(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0 = mogg.routine (%arg0 : !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    inputs (%a)
    outputs (%b)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    %s0 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %a0 = mogg.tensor.create(%a, %s0) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %b0 = mogg.tensor.create(%b, %s0) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %a1 = mogg.trait.create(%a0) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    %b1 = mogg.trait.create(%b0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    mogg.call.opaque @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a1 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
  }
  mo.output %0 : !mo.tensor<[2, 2], f32>
}
