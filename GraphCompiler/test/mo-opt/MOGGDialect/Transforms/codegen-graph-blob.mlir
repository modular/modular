// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-outline-structured-kernel-bodies \
// RUN:   --canonicalize --cse --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-form-graph-blobs="max-calls=100" \
// RUN:   --mogg-reuse-graph-blob-buffers --mogg-lower-graph-blobs \
// RUN:   --canonicalize --mogg-outline-routines --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: FileCheck %s < %t/graph.mojo

// Three declarative calls form one graph blob, which formation, buffer reuse
// and lowering write out as one launch. The kernel takes the routine's
// pointers and builds each call's views inside its guard. The first call's
// fused output stays on the host and crosses as its `device_type`, with the
// bias its epilogue captures inside it. The second and third calls both read
// the first call's buffer, which crosses once.

// CHECK:      def {{__[A-Za-z0-9_]+}}([[A:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[BIAS:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[C:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[D:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[E:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[CTX:[A-Za-z0-9_]+]]: {{.*}}DeviceContext) raises:
// CHECK:        var [[FUSED:[A-Za-z0-9_]+]] = _MoggOutputStoreFusionTensor[
// CHECK:        var [[CONFIG0:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarativeFusedStore.execute_configuration(
// CHECK-NEXT:   var [[CONFIG1:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration(
// CHECK-NEXT:   var [[CONFIG2:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration(
// CHECK-NEXT:   var [[LAUNCH:[A-Za-z0-9_]+]] = _graph_blob.launch_config([[CTX]])
// CHECK-NEXT:   var [[ACT0:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[CONFIG0]].get_grid_dim(), [[CONFIG0]].get_block_dim(), [[LAUNCH]].get_grid_dim(), [[LAUNCH]].get_block_dim())
// CHECK-NEXT:   var [[ACT1:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[CONFIG1]].get_grid_dim(),
// CHECK-NEXT:   var [[ACT2:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[CONFIG2]].get_grid_dim(),
// CHECK-NEXT:   comptime [[A_T:[A-Za-z0-9_]+]] = type_of([[A]])
// CHECK-NEXT:   comptime [[FUSED_T:[A-Za-z0-9_]+]] = type_of([[FUSED]])
// CHECK-NEXT:   comptime [[C_T:[A-Za-z0-9_]+]] = type_of([[C]])
// CHECK-NEXT:   comptime [[D_T:[A-Za-z0-9_]+]] = type_of([[D]])
// CHECK-NEXT:   comptime [[E_T:[A-Za-z0-9_]+]] = type_of([[E]])
// CHECK:        def [[KERNEL:execute_device_[_0-9]+]]([[X:[A-Za-z0-9_]+]]: [[A_T]].device_type, [[O0:[A-Za-z0-9_]+]]: [[FUSED_T]].device_type, [[Y:[A-Za-z0-9_]+]]: [[C_T]].device_type, [[Z0:[A-Za-z0-9_]+]]: [[D_T]].device_type, [[Z1:[A-Za-z0-9_]+]]: [[E_T]].device_type, [[A0:[A-Za-z0-9_]+]]: {{[^,]+}}, [[A1:[A-Za-z0-9_]+]]: {{[^,]+}}, [[A2:[A-Za-z0-9_]+]]: {{[^,]+}}):
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A0]]){{\)?}}:
// CHECK:            mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[X]].unsafe_bitcast
// CHECK:            var [[IN0:[A-Za-z0-9_]+]] = _MoggDenseTensorArg[{{.*}}_MoggTensorIO.INPUT
// CHECK-NEXT:       {{.*}}.MapLowerDeclarativeFusedStore.execute_device([[O0]], [[IN0]])
// CHECK:          _graph_blob.barrier()
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A1]]){{\)?}}:
// CHECK:            mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[Y]].unsafe_bitcast
// CHECK:            mogg_tensor_init[{{.*}}IO.Output{{.*}}]([[Z0]].unsafe_bitcast
// CHECK:            {{.*}}.MapLowerDeclarative.execute_device(
// CHECK:          _graph_blob.barrier()
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A2]]){{\)?}}:
// CHECK:            mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[Y]].unsafe_bitcast
// CHECK:            mogg_tensor_init[{{.*}}IO.Output{{.*}}]([[Z1]].unsafe_bitcast
// CHECK:            {{.*}}.MapLowerDeclarative.execute_device(
// CHECK:        builtin_primitives.primitives.mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}([[CTX]], [[LAUNCH]], [[A]], [[FUSED]], [[C]], [[D]], [[E]], [[ACT0]], [[ACT1]], [[ACT2]])
// CHECK-NOT:    mogg_launch_declarative

mo.graph @blob(%a: !mo.tensor<[2, 2], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) {
  %b = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %c = mo.add(%b, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %e = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %d, %e : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>
}
