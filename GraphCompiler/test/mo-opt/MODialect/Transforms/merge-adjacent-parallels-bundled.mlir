// RUN: mo-opt %s --mo-merge-adjacent-parallels --verify-each | FileCheck %s

// Exercises the multi-producer merge path on the realistic
// bundled-allreduce shape: two `mo.parallel { mo.bundled.allreduce.sum
// }` producers (with signal buffers + chain) and a third chainless
// per-device `mo.add` parallel that fans in from both -- the shape
// that `mo-create-parallel-from-unbundle` produces for a residual add
// after an allreduce.  The merge must pick the second allreduce as
// the consumer's producer (the first one fails the dependence check
// via the chain-dependency carried in through the second allreduce's
// inChain), absorb the `mo.add`, and on a follow-up iteration pull
// in the first allreduce too so the whole chain ends up in a single
// parallel.

// CHECK-LABEL: mo.graph @bundled_allreduce_residual_add
mo.graph @bundled_allreduce_residual_add(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %in_a0: !mo.tensor<[3], f32, gpu:0>, %in_a1: !mo.tensor<[3], f32, gpu:1>,
    %in_b0: !mo.tensor<[3], f32, gpu:0>, %in_b1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>, !mo.chain) {
  %dt_a = mo.tensor.bundle(%in_a0, %in_a1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %dt_b = mo.tensor.bundle(%in_b0, %in_b1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()

  // First allreduce.
  %a_bundle, %a_chain = mo.parallel (%arg, %bch_a) in (%dt_a : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>, %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch) -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %peer0, %peer1 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %sig0, %sig1, %bch_a)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }

  // Second allreduce, sharing signal buffers with the first and taking
  // a chain dependency on the first's out-chain.
  %b_bundle, %b_chain = mo.parallel (%arg, %bch_b) in (%dt_b : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>, %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%a_chain) -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %peer0, %peer1 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %peer1, %sig0, %sig1, %bch_b)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }

  // Per-device residual add, already wrapped in its own chainless
  // mo.parallel (matching what mo-create-parallel-from-unbundle
  // produces for an external per-device consumer of two unbundled
  // bundle outputs).
  %add_bundle = mo.parallel (%arg_a, %arg_b) in (%a_bundle
        : !mo.bundle<[3], f32, gpu:0-1>,
                                                 %b_bundle
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_b) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %s : !mo.tensor<[3], f32, gpu:0>
  }

  // Two iterations should collapse the chain into a single
  // mo.parallel containing both allreduces and the residual add.
  // CHECK:     mo.parallel
  // CHECK:       mo.bundled.allreduce.sum
  // CHECK:       mo.bundled.allreduce.sum
  // CHECK:       mo.add
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %add_bundle, %b_chain
      : !mo.bundle<[3], f32, gpu:0-1>, !mo.chain
}
