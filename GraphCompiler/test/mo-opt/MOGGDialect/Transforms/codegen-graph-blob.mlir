// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo
//
// The hand-written `mogg.call.execute_device`s name kernel declarations that
// only exist once `--mogg-import-packages` has run, so parse-time
// verification is disabled.

// A `mogg.graph_blob` launches its declarative calls as one kernel: each
// call's `execute_configuration` runs on the host, the blob launches their
// `merge`, and the kernel runs each `execute_device` only where
// `is_thread_active` accepts that call's `active_threads`, with a barrier
// between them. The intermediate `%b` crosses the launch once per use.

// CHECK-NOT: import max._graph_blob
// CHECK: def {{__[A-Za-z0-9_]+}}({{.*}}, [[CTX:[A-Za-z0-9_]+]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[CONFIG0:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration([[OUT0:[A-Za-z0-9_]+]], [[IN0:[A-Za-z0-9_]+]])
// CHECK:   comptime [[OUT0_T:[A-Za-z0-9_]+]] = type_of([[OUT0]])
// CHECK:   comptime [[IN0_T:[A-Za-z0-9_]+]] = type_of([[IN0]])
// CHECK:   var [[CONFIG1:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration([[OUT1:[A-Za-z0-9_]+]], [[IN1:[A-Za-z0-9_]+]])
// CHECK:   comptime [[OUT1_T:[A-Za-z0-9_]+]] = type_of([[OUT1]])
// CHECK:   comptime [[IN1_T:[A-Za-z0-9_]+]] = type_of([[IN1]])
// CHECK:   comptime [[CONFIG_T:[A-Za-z0-9_]+]] = type_of([[CONFIG0]])
// CHECK:   var [[CONFIG:[A-Za-z0-9_]+]] = [[CONFIG0]].merge([[CONFIG1]])
// CHECK:   def [[KERNEL:execute_device_[_0-9]+]]([[A0:[A-Za-z0-9_]+]]: [[OUT0_T]].TileTensorType, [[A1:[A-Za-z0-9_]+]]: [[IN0_T]].TileTensorType, [[A2:[A-Za-z0-9_]+]]: [[OUT1_T]].TileTensorType, [[A3:[A-Za-z0-9_]+]]: [[IN1_T]].TileTensorType, [[ACTIVE0:[A-Za-z0-9_]+]]: [[CONFIG_T]].ActiveThreads, [[ACTIVE1:[A-Za-z0-9_]+]]: [[CONFIG_T]].ActiveThreads):
// CHECK-NEXT: if {{\(?}}[[CONFIG_T]].is_thread_active([[ACTIVE0]]){{\)?}}:
// CHECK-NEXT: {{.*}}.MapLowerDeclarative.execute_device([[OUT0_T]]([[A0]]), [[IN0_T]]([[A1]]))
// CHECK: [[CONFIG_T]].barrier()
// CHECK-NEXT: if {{\(?}}[[CONFIG_T]].is_thread_active([[ACTIVE1]]){{\)?}}:
// CHECK-NEXT: {{.*}}.MapLowerDeclarative.execute_device([[OUT1_T]]([[A2]]), [[IN1_T]]([[A3]]))
// CHECK:   builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}([[CTX]], [[CONFIG]], [[OUT0]].to_tile_tensor(), [[IN0]].to_tile_tensor(), [[OUT1]].to_tile_tensor(), [[IN1]].to_tile_tensor(), [[CONFIG0]].active_threads(), [[CONFIG1]].active_threads())

mo.graph @blob(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
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
      mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a1 : !mogg.tensor) outputs(%b1 : !mogg.tensor) context(%ctx : !mogg.context)
      %s2 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %b2 = mogg.tensor.create(%b, %s2) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %s3 = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
      %c0 = mogg.tensor.create(%c, %s3) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
      %b3 = mogg.trait.create(%b2) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      %c1 = mogg.trait.create(%c0) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
      mogg.call.execute_device @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b3 : !mogg.tensor) outputs(%c1 : !mogg.tensor) context(%ctx : !mogg.context)
    }
  }
  mo.output %0#1 : !mo.tensor<[2, 2], f32>
}
