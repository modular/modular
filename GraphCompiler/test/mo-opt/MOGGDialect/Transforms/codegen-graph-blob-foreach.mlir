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

// A generated elementwise kernel between two declarative calls joins their
// blob. Its loop runs in the kernel between the barriers, on the whole launch:
// its active count comes from the launch's own configuration. Its captures are
// built inside its guard from the routine's pointers, as a declarative call's
// operands are, and the registered `_graph_blob_foreach` picks its width from
// the body's dtypes.

// CHECK:      def {{__[A-Za-z0-9_]+}}([[A:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[BIAS:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[B:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[C:[A-Za-z0-9_]+]]: OpaquePointer[MutAnyOrigin], [[CTX:[A-Za-z0-9_]+]]: {{.*}}DeviceContext) raises:
// CHECK:        var [[CONFIG0:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration(
// CHECK-NEXT:   var [[CONFIG2:[A-Za-z0-9_]+]] = {{.*}}.MapLowerDeclarative.execute_configuration(
// CHECK-NEXT:   var [[LAUNCH:[A-Za-z0-9_]+]] = _graph_blob.launch_config([[CTX]])
// CHECK-NEXT:   var [[ACT0:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[CONFIG0]].get_grid_dim(),
// CHECK-NEXT:   var [[ACT1:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[LAUNCH]].get_grid_dim(), [[LAUNCH]].get_block_dim(), [[LAUNCH]].get_grid_dim(), [[LAUNCH]].get_block_dim())
// CHECK-NEXT:   var [[ACT2:[A-Za-z0-9_]+]] = _graph_blob.active_threads([[CONFIG2]].get_grid_dim(),
// CHECK:        def [[KERNEL:execute_device_[_0-9]+]]([[X:[A-Za-z0-9_]+]]: {{[^,]+}}, [[Y:[A-Za-z0-9_]+]]: {{[^,]+}}, [[Z:[A-Za-z0-9_]+]]: {{[^,]+}}, [[W:[A-Za-z0-9_]+]]: {{[^,]+}}, [[A0:[A-Za-z0-9_]+]]: {{[^,]+}}, [[A1:[A-Za-z0-9_]+]]: {{[^,]+}}, [[A2:[A-Za-z0-9_]+]]: {{[^,]+}}):
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A0]]){{\)?}}:
// CHECK:            .MapLowerDeclarative.execute_device(
// CHECK:          _graph_blob.barrier()
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A1]]){{\)?}}:
// CHECK-NEXT:       var [[SHAPE:[A-Za-z0-9_]+]] = layout.Coord(layout.Idx[2], layout.Idx[2])
// CHECK:            mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[Y]].unsafe_bitcast
// CHECK-NEXT:       mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[Z]].unsafe_bitcast
// CHECK-NEXT:       mogg_tensor_init[{{.*}}IO.Output{{.*}}]([[W]].unsafe_bitcast
// CHECK:            def [[LAMBDA:lambda_[_0-9]+]][width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK:              Add.elementwise
// CHECK:            builtin_primitives.primitives._graph_blob_foreach[dtypes = std.utils.StaticTuple[DType, 1](DType.float32)]([[SHAPE]], [[LAMBDA]])
// CHECK:          _graph_blob.barrier()
// CHECK-NEXT:     if {{\(?}}_graph_blob.is_thread_active([[A2]]){{\)?}}:
// CHECK:            mogg_tensor_init[{{.*}}IO.Input{{.*}}]([[W]].unsafe_bitcast
// CHECK:            mogg_tensor_init[{{.*}}IO.Output{{.*}}]([[Y]].unsafe_bitcast
// CHECK:            .MapLowerDeclarative.execute_device(
// CHECK:        mogg_launch_declarative{{\[}}[[KERNEL]]{{\]}}([[CTX]], [[LAUNCH]], [[A]], [[B]], [[BIAS]], [[C]], [[ACT0]], [[ACT1]], [[ACT2]])
// CHECK-NOT:    mogg_launch_declarative

mo.graph @blob(%a: !mo.tensor<[2, 2], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %c = mo.add(%b, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %d : !mo.tensor<[2, 2], f32, gpu:0>
}
