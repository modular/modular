// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mogg-outline-routines --mogg-emit-mojo -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo
//
// The calls name kernel declarations that only exist once
// `--mogg-import-packages` has run, so parse-time verification is disabled.

// Each op of a lowered graph blob prints its own Mojo: the host ops bind the
// configurations, the launch configuration and the active counts; the launch
// declares one kernel parameter per operand and passes the operands; each call
// region builds its views inside the kernel and runs `execute_device` under
// its guard, with the barrier between them.

// CHECK: import max._graph_blob as _graph_blob

// CHECK:      def {{__[A-Za-z0-9_]+}}([[A:[A-Za-z0-9_]+]]: {{[^,]+}}, [[B:[A-Za-z0-9_]+]]: {{[^,]+}}, [[C:[A-Za-z0-9_]+]]: {{[^,]+}}, [[CTX:[A-Za-z0-9_]+]]: {{.*}}DeviceContext) raises:
// CHECK:        var [[CONFIG0:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration({{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}})
// CHECK:        var [[CONFIG1:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration({{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}})
// CHECK-NEXT:   var [[LAUNCH:[A-Za-z0-9_]+]] = _graph_blob.launch_config([[CTX]])
// CHECK-NEXT:   var [[ACT0:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[CONFIG0]].get_grid_dim(), [[CONFIG0]].get_block_dim(), [[LAUNCH]].get_grid_dim(), [[LAUNCH]].get_block_dim())
// CHECK-NEXT:   var [[ACT1:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[CONFIG1]].get_grid_dim(), [[CONFIG1]].get_block_dim(), [[LAUNCH]].get_grid_dim(), [[LAUNCH]].get_block_dim())
// CHECK-NEXT:   comptime [[A_T:[A-Za-z0-9_]+]] = type_of([[A]])
// CHECK-NEXT:   comptime [[B_T:[A-Za-z0-9_]+]] = type_of([[B]])
// CHECK-NEXT:   comptime [[C_T:[A-Za-z0-9_]+]] = type_of([[C]])
// CHECK-NEXT:   comptime [[ACT0_T:[A-Za-z0-9_]+]] = type_of([[ACT0]])
// CHECK-NEXT:   comptime [[ACT1_T:[A-Za-z0-9_]+]] = type_of([[ACT1]])
// CHECK-NEXT:   def [[KERNEL:execute_device_[_0-9]+]]([[X:[A-Za-z0-9_]+]]: [[A_T]].device_type, [[Y:[A-Za-z0-9_]+]]: [[B_T]].device_type, [[Z:[A-Za-z0-9_]+]]: [[C_T]].device_type, [[A0:[A-Za-z0-9_]+]]: [[ACT0_T]].device_type, [[A1:[A-Za-z0-9_]+]]: [[ACT1_T]].device_type):
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A0]]){{\)?}}:
// CHECK:            mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[X]].unsafe_bitcast
// CHECK:            mogg_tensor_init[{{.*}}IO.Output{{.*}}]([[Y]].unsafe_bitcast
// CHECK:            var [[IN0:[A-Za-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.INPUT
// CHECK:            var [[OUT0:[A-Za-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.OUTPUT
// CHECK-NEXT:       {{.*}}.MapLowerDeclarative.execute_device([[OUT0]], [[IN0]])
// CHECK:          _graph_blob.barrier()
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A1]]){{\)?}}:
// CHECK:            mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[Y]].unsafe_bitcast
// CHECK:            mogg_tensor_init[{{.*}}IO.Output{{.*}}]([[Z]].unsafe_bitcast
// CHECK:            var [[IN1:[A-Za-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.INPUT
// CHECK:            var [[OUT1:[A-Za-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.OUTPUT
// CHECK-NEXT:       {{.*}}.MapLowerDeclarative.execute_device([[OUT1]], [[IN1]])
// CHECK:        builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}([[CTX]], [[LAUNCH]], [[A]], [[B]], [[C]], [[ACT0]], [[ACT1]])
mo.graph @blob(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %0:2 = mogg.routine (%arg0 : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
    inputs (%a)
    outputs (%b, %c)
    context (%ctx) attributes {device = #M.device_ref<"gpu", 0>} {
    %s = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
    %a_in = mogg.tensor.create(%a, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %b_out = mogg.tensor.create(%b, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %a_trait = mogg.trait.create(%a_in) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    %b_trait = mogg.trait.create(%b_out) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    %cfg0 = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a_trait : !mogg.tensor) outputs(%b_trait : !mogg.tensor)
    %b_in = mogg.tensor.create(%b, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %c_out = mogg.tensor.create(%c, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
    %b_in_trait = mogg.trait.create(%b_in) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    %c_trait = mogg.trait.create(%c_out) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    %cfg1 = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%b_in_trait : !mogg.tensor) outputs(%c_trait : !mogg.tensor)
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

// A dim crosses as an `Int64` kernel argument, which the kernel reads as an
// `Int`.

// CHECK:      def {{__[A-Za-z0-9_]+}}({{[^,]+}}, {{[^,]+}}, [[N:[A-Za-z0-9_]+]]: Int, [[CTX:[A-Za-z0-9_]+]]: {{.*}}DeviceContext) raises:
// CHECK:        def [[KERNEL:execute_device_[_0-9]+]]({{[A-Za-z0-9_]+}}: {{[A-Za-z0-9_]+}}.device_type, {{[A-Za-z0-9_]+}}: {{[A-Za-z0-9_]+}}.device_type, [[D:[A-Za-z0-9_]+]]: Int64, {{[A-Za-z0-9_]+}}: {{[A-Za-z0-9_]+}}.device_type):
// CHECK:          layout.Coord({{.*}}Int([[D]]){{.*}})
// CHECK:        builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}([[CTX]], {{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}, {{[A-Za-z0-9_]+}}, Int64([[N]]), {{[A-Za-z0-9_]+}})
mo.graph @blob_dim<batch>(%arg0: !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32> {
  %0 = mogg.routine (%arg0 : !mo.tensor<[batch, 2], f32>) -> !mo.tensor<[batch, 2], f32>
    inputs (%a)
    outputs (%b)
    context (%ctx)
    dims (%n : !mogg.dim<batch>) attributes {device = #M.device_ref<"gpu", 0>} {
    %s = mogg.shape.create(%n) <shape = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.dim<batch>
    %a_in = mogg.tensor.create(%a, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
    %b_out = mogg.tensor.create(%b, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[batch, 2]> : !mosh.ape> : !mogg.tensor
    %a_trait = mogg.trait.create(%a_in) contract = <{role = input, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    %b_trait = mogg.trait.create(%b_out) contract = <{role = output, fused = false, access = none, capabilities = none}>, device = "gpu" : (!mogg.tensor) -> !mogg.tensor
    %cfg = mogg.call.execute_configuration @testing_kernels.testing_kernels.MapLowerDeclarative inputs(%a_trait : !mogg.tensor) outputs(%b_trait : !mogg.tensor)
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
