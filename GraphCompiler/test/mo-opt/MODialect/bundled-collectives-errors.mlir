// RUN: mo-opt %s --split-input-file --verify-diagnostics

// A bundled collective outside mo.parallel is rejected in a plain graph and
// accepted inside an isDeviceGraph graph or a mo.parallel.

// A bundled collective directly in a plain (non-device) graph body.
mo.graph @no_parallel_no_device_graph(
    %a: !mo.tensor<[3], f32, gpu:0>,
    %b: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain) {
  %ch = mo.chain.create()
  // expected-error @below {{'mo.bundled.allreduce.sum' op must be inside a mo.parallel region or a mo.graph with the isDeviceGraph attribute}}
  %out, %och = mo.bundled.allreduce.sum(%a, %b, %sig0, %sig1, %ch)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// -----

// The same collective is valid in an isDeviceGraph graph, the form
// mo-mark-stable-tensors produces.
mo.graph @device_graph(
    %a: !mo.tensor<[3], f32, gpu:0>,
    %b: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  %out, %och = mo.bundled.allreduce.sum(%a, %b, %sig0, %sig1, %ch)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// -----

// The same collective is valid inside mo.parallel in a plain graph, taking
// peer tensors from mo.bundled.expand and signal buffers from the parallel's
// buffers(...) clause.
mo.graph @parallel(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:1>
  %bundle = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %ar:2 = mo.parallel (%arg, %bch) in (%bundle : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %p:2 = mo.bundled.expand(%arg)
        : <[3], f32, gpu:0>
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %och = mo.bundled.allreduce.sum(%p#0, %p#1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
           !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %och : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u0, %u1 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  mo.output %u0, %ar#1 : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// -----

// A grouped copy with an explicit rank, the form mo-mark-stable-tensors
// produces for a grouped `mo.distributed.allreduce.sum`: every copy takes the
// whole world, and sibling groups may differ in shape.
mo.graph @grouped_with_rank(
    %a: !mo.tensor<[3], f32, gpu:0>, %b: !mo.tensor<[3], f32, gpu:1>,
    %c: !mo.tensor<[5], f32, gpu:2>, %d: !mo.tensor<[5], f32, gpu:3>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>, %sig3: !mo.buffer<[1], ui8, gpu:3>
) -> (!mo.tensor<[5], f32, gpu:2>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  %out, %och = mo.bundled.allreduce.sum(%a, %b, %c, %d, %sig0, %sig1, %sig2, %sig3, %ch) <group_size = 2, peer_rank = 2>
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.tensor<[5], f32, gpu:2>, !mo.tensor<[5], f32, gpu:3>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain)
     -> (!mo.tensor<[5], f32, gpu:2>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[5], f32, gpu:2>, !mo.chain
}

// -----

mo.graph @group_size_does_not_divide(
    %a: !mo.tensor<[3], f32, gpu:0>, %b: !mo.tensor<[3], f32, gpu:1>,
    %c: !mo.tensor<[3], f32, gpu:2>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{group_size must evenly divide the number of inputs (3), but got 2}}
  %out, %och = mo.bundled.allreduce.sum(%a, %b, %c, %sig0, %sig1, %sig2, %ch) <group_size = 2>
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:2>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// -----

// Shapes must agree within a group; here the two members of group 0 differ.
mo.graph @group_shape_mismatch(
    %a: !mo.tensor<[3], f32, gpu:0>, %b: !mo.tensor<[4], f32, gpu:1>,
    %c: !mo.tensor<[5], f32, gpu:2>, %d: !mo.tensor<[5], f32, gpu:3>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>, %sig3: !mo.buffer<[1], ui8, gpu:3>
) -> (!mo.tensor<[5], f32, gpu:2>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{all inputs in each allreduce group must have the same shape}}
  %out, %och = mo.bundled.allreduce.sum(%a, %b, %c, %d, %sig0, %sig1, %sig2, %sig3, %ch) <group_size = 2, peer_rank = 2>
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[5], f32, gpu:2>, !mo.tensor<[5], f32, gpu:3>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain)
     -> (!mo.tensor<[5], f32, gpu:2>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[5], f32, gpu:2>, !mo.chain
}

// -----

mo.graph @peer_rank_out_of_range(
    %a: !mo.tensor<[3], f32, gpu:0>, %b: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{peer_rank 2 out of range for 2 devices}}
  %out, %och = mo.bundled.allreduce.sum(%a, %b, %sig0, %sig1, %ch) <peer_rank = 2>
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// -----

// `peer_rank` indexes the signal buffers, so it must name this device's.
mo.graph @peer_rank_wrong_device(
    %a: !mo.tensor<[3], f32, gpu:0>, %b: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{peer_rank 1 names the signal buffer on #M.device_ref<"gpu", 1>, but the op runs on #M.device_ref<"gpu", 0>}}
  %out, %och = mo.bundled.allreduce.sum(%a, %b, %sig0, %sig1, %ch) <peer_rank = 1>
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// -----

// The same rank check applies to the other bundled collectives.
mo.graph @broadcast_peer_rank_wrong_device(
    %a: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{peer_rank 1 names the signal buffer on #M.device_ref<"gpu", 1>, but the op runs on #M.device_ref<"gpu", 0>}}
  %out, %och = mo.bundled.broadcast(%a, %sig0, %sig1, %ch) <root = 1, peer_rank = 1>
      : (!mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
  mo.output %out, %och : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// -----

mo.graph @allgather_peer_rank_out_of_range(
    %a: !mo.tensor<[3], f32, gpu:0>, %b: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error @below {{peer_rank -1 out of range for 2 devices}}
  %o0, %o1, %och = mo.bundled.allgather(%a, %b, %sig0, %sig1, %ch) <peer_rank = -1>
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
         !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.chain)
  mo.output %o0, %o1, %och : !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.chain
}
