// RUN: mo-opt --split-input-file --verify-diagnostics \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-mark-stable-tensors %s

// The op's operands must map onto the kernel's `build_graph` signature before
// any stable position can be named; a mismatch is reported on the op.
mo.graph @too_many_operands(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    attributes {isDeviceGraph} {
  // expected-error @below {{cannot map operands onto the `build_graph` function of kernel}}
  %0 = mo.custom [symbol = "imposter_stable_single_build_graph", device = #M.device_ref<"gpu", 0>]
    (%arg0, %arg0) : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// A grouped allgather routes part of a shard through a paired group's
// buffers, which a device graph cannot express.
mo.graph @grouped_allgather(
    %arg0: !mo.tensor<[2], f32, gpu:0>, %arg1: !mo.tensor<[2], f32, gpu:1>,
    %arg2: !mo.tensor<[2], f32, gpu:2>, %arg3: !mo.tensor<[2], f32, gpu:3>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>, %sig3: !mo.buffer<[1], ui8, gpu:3>
) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[2], f32, gpu:2>, !mo.tensor<[2], f32, gpu:2>, !mo.tensor<[2], f32, gpu:3>, !mo.tensor<[2], f32, gpu:3>)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{a device graph supports only a full-world allgather (group_size equal to the number of inputs, 4), but group_size is 2}}
  %0, %1, %2, %3, %4, %5, %6, %7, %och = mo.distributed.allgather(%arg0, %arg1, %arg2, %arg3, %sig0, %sig1, %sig2, %sig3, %ch) [group_size = 2]
    : (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[2], f32, gpu:2>, !mo.tensor<[2], f32, gpu:3>,
       !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain)
    -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[2], f32, gpu:2>, !mo.tensor<[2], f32, gpu:2>, !mo.tensor<[2], f32, gpu:3>, !mo.tensor<[2], f32, gpu:3>, !mo.chain)
  mo.output %0, %1, %2, %3, %4, %5, %6, %7 : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[2], f32, gpu:2>, !mo.tensor<[2], f32, gpu:2>, !mo.tensor<[2], f32, gpu:3>, !mo.tensor<[2], f32, gpu:3>
}

// -----

// A row-sliced allgather has no bundled form either.
mo.graph @sliced_allgather(
    %arg0: !mo.tensor<[4, 2], f32, gpu:0>, %arg1: !mo.tensor<[4, 2], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>, !mo.tensor<[2, 2], f32, gpu:1>)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{a device graph has no per-device form of a row-sliced allgather (slice_rows)}}
  %0, %1, %2, %3, %och = mo.distributed.allgather(%arg0, %arg1, %sig0, %sig1, %ch) [group_size = 2, slice_rows = true]
    : (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[4, 2], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>, !mo.tensor<[2, 2], f32, gpu:1>, !mo.chain)
  mo.output %0, %1, %2, %3 : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>, !mo.tensor<[2, 2], f32, gpu:1>
}
