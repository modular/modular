// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels --map-allocate-accumulating-buffers \
// RUN:   --map-raise-back-contiguous-views --map-outline-structured-kernel-bodies \
// RUN:   --map-fuse-contiguous-views --canonicalize --cse --map-to-mogg \
// RUN:   --mogg-conform-call-operands --mogg-form-graph-blobs \
// RUN:   --mogg-reuse-graph-blob-buffers | FileCheck %s

// Members 0 to 3 run a -> b -> c -> d -> e, and the graph returns e. b is
// free after member 1, so d takes its buffer; c is free after member 2, so e
// takes its buffer. The routine returns e from c's buffer.
// CHECK-LABEL: mo.graph @sequence
// CHECK:       %[[R:.*]]:2 = mogg.routine (%{{.*}} : !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
// CHECK-NEXT:    inputs (%[[A:[^ ,)]+]])
// CHECK-NEXT:    outputs (%[[P0:[^ ,)]+]], %[[P1:[^ ,)]+]])
// CHECK:         mogg.graph_blob {
// CHECK:           mogg.tensor.create(%[[A]],
// CHECK:           mogg.tensor.create(%[[P0]],
// CHECK:           mogg.call.execute_device
// CHECK:           mogg.tensor.create(%[[P0]],
// CHECK:           mogg.tensor.create(%[[P1]],
// CHECK:           mogg.call.execute_device
// CHECK:           mogg.tensor.create(%[[P1]],
// CHECK:           mogg.tensor.create(%[[P0]],
// CHECK:           mogg.call.execute_device
// CHECK:           mogg.tensor.create(%[[P0]],
// CHECK:           mogg.tensor.create(%[[P1]],
// CHECK:           mogg.call.execute_device
// CHECK:       mo.output %[[R]]#1
mo.graph @sequence(%a: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %e = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%d) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %e : !mo.tensor<[2, 2], f32>
}

// Members 1 and 2 both read b, so b is busy until member 2, and c and d are
// both read outside the routine. No buffer is free when an output needs one.
// CHECK-LABEL: mo.graph @fan_out
// CHECK:       %[[R:.*]]:3 = mogg.routine
// CHECK:       mo.output %[[R]]#1, %[[R]]#2
mo.graph @fan_out(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %c, %d : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// The graph returns b, so its buffer stays b's to the end of the routine.
// CHECK-LABEL: mo.graph @shared_intermediate
// CHECK:       %[[R:.*]]:2 = mogg.routine
// CHECK:       mo.output %[[R]]#0, %[[R]]#1
mo.graph @shared_intermediate(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %b, %c : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// b is free after member 1, when member 2 writes d, but d has another shape,
// so every output keeps its buffer, though both hold 16 bytes.
// CHECK-LABEL: mo.graph @same_size_different_shape
// CHECK:       %[[R:.*]]:4 = mogg.routine
// CHECK:       mo.output %[[R]]#1, %[[R]]#3
mo.graph @same_size_different_shape(%a: !mo.tensor<[2, 2], f32>, %y: !mo.tensor<[4], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%y) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %e = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%d) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %c, %e : !mo.tensor<[2, 2], f32>, !mo.tensor<[4], f32>
}

// A routine that updates a buffer in place keeps its chain: b is free after
// member 1, so d takes its buffer, and the routine still returns the blob's
// chain.
// CHECK-LABEL: mo.graph @buffer_chain
// CHECK:       %[[R:.*]]:3 = mogg.routine (%{{.*}}, %{{.*}} : !mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>) chain(%{{.*}}) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.chain)
// CHECK:       mo.output %[[R]]#0
mo.graph @buffer_chain(%a: !mo.tensor<[2, 2], f32>, %cache: !mo.buffer<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %ch0 = mo.chain.create()
  %b, %ch1 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%a, %cache, %ch0) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %ch2 = mo.chain.create(%ch1, %ch0) : !mo.chain, !mo.chain
  %c, %ch3 = mo.custom [symbol = "map_lower_declarative_inplace", device = #M.device_ref<"gpu", 0>](%b, %cache, %ch2) : (!mo.tensor<[2, 2], f32>, !mo.buffer<[2, 2], f32>, !mo.chain) -> (!mo.tensor<[2, 2], f32>, !mo.chain)
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %d : !mo.tensor<[2, 2], f32>
}

// b is free after member 1, when member 2 writes d, but d is larger, so every
// output keeps its buffer.
// CHECK-LABEL: mo.graph @different_sizes
// CHECK:       %[[R:.*]]:4 = mogg.routine
// CHECK:       mo.output %[[R]]#1, %[[R]]#3
mo.graph @different_sizes(%a: !mo.tensor<[2, 2], f32>, %y: !mo.tensor<[8, 8], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[8, 8], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%y) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %e = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%d) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  mo.output %c, %e : !mo.tensor<[2, 2], f32>, !mo.tensor<[8, 8], f32>
}

// Member 3 reads b through a reshape fused into its kernel, so b stays busy
// until member 3 and d, written by member 2, gets a buffer of its own.
// CHECK-LABEL: mo.graph @view_read_later
// CHECK:       %[[R:.*]]:4 = mogg.routine
// CHECK:       mo.output %[[R]]#2, %[[R]]#3
mo.graph @view_read_later(%a: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[4], f32>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %r = mo.static.reshape(%b) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[4], f32>
  %e = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%r) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %d, %e : !mo.tensor<[2, 2], f32>, !mo.tensor<[4], f32>
}

// Members whose outputs go through a fused epilogue share buffers like any
// others: the chain needs two.
// CHECK-LABEL: mo.graph @fused_store_chain
// CHECK:       %[[R:.*]]:2 = mogg.routine
// CHECK:       mo.output %[[R]]#1
mo.graph @fused_store_chain(%a: !mo.tensor<[2, 2], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %x1 = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %y1 = mo.add(%x1, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %x2 = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%y1) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %y2 = mo.add(%x2, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %x3 = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%y2) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %y3 = mo.add(%x3, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %x4 = mo.custom [symbol = "map_lower_declarative_fused_store", device = #M.device_ref<"gpu", 0>](%y3) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %y4 = mo.add(%x4, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %y4 : !mo.tensor<[2, 2], f32, gpu:0>
}

// Member 3 reads b through a multiply fused into its input, so b stays busy
// until member 3. d, written by member 2, gets a buffer of its own, and e
// takes c's.
// CHECK-LABEL: mo.graph @fused_load_read_later
// CHECK:       %[[R:.*]]:3 = mogg.routine
// CHECK:       mo.output %[[R]]#2, %[[R]]#1
mo.graph @fused_load_read_later(%a: !mo.tensor<[2, 2], f32, gpu:0>, %bias: !mo.tensor<[2, 2], f32, gpu:0>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>) {
  %b = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%a) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %c = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%b) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %d = mo.custom [symbol = "map_lower_declarative", device = #M.device_ref<"gpu", 0>](%c) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  %x = mo.mul(%b, %bias) : !mo.tensor<[2, 2], f32, gpu:0>
  %e = mo.custom [symbol = "map_lower_declarative_fused_load", device = #M.device_ref<"gpu", 0>](%x) : (!mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %d, %e : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>
}
