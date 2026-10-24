// RUN: mo-opt %s --mo-create-parallel-from-unbundle --canonicalize | FileCheck %s

// The `--canonicalize` step is part of the pipeline: the pass emits a
// transient `bundle(unbundle(X)...)` that canonicalize folds back to `X`,
// which is what `mo-merge-adjacent-parallels` matches on downstream.

// ============================================================================
// Unary consumer: wrap one relu per launch.
// ============================================================================

// CHECK-LABEL: mo.graph @wrap_unary_relu
mo.graph @wrap_unary_relu(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:1>
  %b = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %ar:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %p:2 = mo.bundled.expand(%arg)
        : <[3], f32, gpu:0>
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%p#0, %p#1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
           !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u:2 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %r0 = mo.relu(%u#0) : !mo.tensor<[3], f32, gpu:0>
  %r1 = mo.relu(%u#1) : !mo.tensor<[3], f32, gpu:1>

  // The allreduce parallel is untouched; a fresh parallel appears adjacent,
  // taking the allreduce's bundle output directly (after canonicalize
  // folded the transient bundle/unbundle pair away).
  // CHECK: %[[AR:[a-zA-Z0-9_]+]]:2 = mo.parallel
  // CHECK:   mo.bundled.allreduce.sum
  // CHECK: mo.parallel ({{.*}}) in (%[[AR]]#0
  // CHECK:   mo.relu
  // CHECK:   mo.yield

  %t0, %tc0 = mo.transfer[%ar#1] %r0
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %t1, %tc1 = mo.transfer[%tc0] %r1
      : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %t0, %t1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// ============================================================================
// Binary consumer with per-launch residuals: driving bundle input plus
// one BUNDLE-classified residual input.
// ============================================================================

// CHECK-LABEL: mo.graph @wrap_binary_add
mo.graph @wrap_binary_add(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %r0: !mo.tensor<[3], f32, gpu:0>,
    %r1: !mo.tensor<[3], f32, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:1>
  %b = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %ar:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %p:2 = mo.bundled.expand(%arg)
        : <[3], f32, gpu:0>
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%p#0, %p#1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
           !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u:2 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %s0 = mo.add(%u#0, %r0) : !mo.tensor<[3], f32, gpu:0>
  %s1 = mo.add(%u#1, %r1) : !mo.tensor<[3], f32, gpu:1>

  // A second parallel wraps the add.  Input #0 is the allreduce bundle
  // (via SSA equality after canonicalize); input #1 is a fresh residual
  // bundle.
  // CHECK: %[[AR:[a-zA-Z0-9_]+]]:2 = mo.parallel
  // CHECK: %[[RB:[a-zA-Z0-9_]+]] = mo.tensor.bundle(%arg2, %arg3)
  // CHECK: mo.parallel ({{.*}}, {{.*}}) in (%[[AR]]#0 {{.*}} %[[RB]]
  // CHECK:   mo.add
  // CHECK:   mo.yield

  %t0, %tc0 = mo.transfer[%ar#1] %s0
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %t1, %tc1 = mo.transfer[%tc0] %s1
      : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %t0, %t1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// ============================================================================
// Two-step `relu -> relu` chain: the first wrap's output unbundle seeds
// the second wrap, exercising the fixed-point loop.
// ============================================================================

// CHECK-LABEL: mo.graph @wrap_chain_two_relu
mo.graph @wrap_chain_two_relu(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:1>
  %b = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %ar:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %p:2 = mo.bundled.expand(%arg)
        : <[3], f32, gpu:0>
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%p#0, %p#1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
           !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u:2 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %r0a = mo.relu(%u#0) : !mo.tensor<[3], f32, gpu:0>
  %r1a = mo.relu(%u#1) : !mo.tensor<[3], f32, gpu:1>
  %r0b = mo.relu(%r0a) : !mo.tensor<[3], f32, gpu:0>
  %r1b = mo.relu(%r1a) : !mo.tensor<[3], f32, gpu:1>

  // Allreduce parallel untouched, then two fresh parallels appear in
  // sequence (proves the fixed-point loop fired more than once).
  // CHECK: %[[AR:[a-zA-Z0-9_]+]]:2 = mo.parallel
  // CHECK:   mo.bundled.allreduce.sum
  // CHECK: mo.parallel ({{.*}}) in (%[[AR]]#0
  // CHECK:   mo.relu
  // CHECK:   mo.yield
  // CHECK: mo.parallel
  // CHECK:   mo.relu
  // CHECK:   mo.yield

  %t0, %tc0 = mo.transfer[%ar#1] %r0b
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %t1, %tc1 = mo.transfer[%tc0] %r1b
      : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %t0, %t1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// ============================================================================
// Rejection: a single binary consumer that takes both unbundled elements
// (would be nominated as the peer for two launches and cloned/erased
// twice).
// ============================================================================

// CHECK-LABEL: mo.graph @reject_shared_binary_consumer
mo.graph @reject_shared_binary_consumer(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:0>
) -> (!mo.tensor<[3], f32>) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %b = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>)
     -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>]>
  %ch = mo.chain.create()
  // Body has a real op (relu) so the allreduce parallel survives
  // canonicalization and anchors the CHECK below.
  %ar:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>]>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:0>)
      chain(%ch)
      -> (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>]>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r, %bch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u:2 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>]>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>)

  // %add consumes BOTH unbundled elements; the peer-uniqueness check
  // rejects the candidate and the add stays outside any parallel.
  // CHECK: mo.parallel
  // CHECK:   mo.relu
  // CHECK-NOT: mo.parallel
  // CHECK: mo.add
  %add = mo.add(%u#0, %u#1) : !mo.tensor<[3], f32, gpu:0>

  %t0, %tc0 = mo.transfer[%ar#1] %add
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  mo.output %t0 : !mo.tensor<[3], f32>
}

// ============================================================================
// Rejection: a consumer with a shared outer host scalar (would require
// capturing an outer SSA value into the body).  Per-launch
// `mo.reduce.layer_norm` ops share `%eps`, so the `all_equal` check in
// planInputs fires.
// ============================================================================

// CHECK-LABEL: mo.graph @reject_shared_outer_scalar
mo.graph @reject_shared_outer_scalar(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %gamma0: !mo.tensor<[3], f32, gpu:0>,
    %gamma1: !mo.tensor<[3], f32, gpu:1>,
    %beta0: !mo.tensor<[3], f32, gpu:0>,
    %beta1: !mo.tensor<[3], f32, gpu:1>,
    %eps: !mo.tensor<[], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:1>
  %b = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %ar:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %p:2 = mo.bundled.expand(%arg)
        : <[3], f32, gpu:0>
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%p#0, %p#1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
           !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u:2 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)

  // Operand 0 is DRIVING; operands 1 and 2 are per-launch gamma and beta
  // (BUNDLE); operand 3 (%eps) is shared, triggering rejection.
  // CHECK: mo.parallel
  // CHECK:   mo.bundled.allreduce.sum
  // CHECK-NOT: mo.parallel
  // CHECK: mo.reduce.layer_norm
  // CHECK: mo.reduce.layer_norm
  %n0 = mo.reduce.layer_norm(%u#0, %gamma0, %beta0, %eps)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>,
         !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[], f32>)
     -> !mo.tensor<[3], f32, gpu:0>
  %n1 = mo.reduce.layer_norm(%u#1, %gamma1, %beta1, %eps)
      : (!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:1>,
         !mo.tensor<[3], f32, gpu:1>, !mo.tensor<[], f32>)
     -> !mo.tensor<[3], f32, gpu:1>

  %t0, %tc0 = mo.transfer[%ar#1] %n0
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %t1, %tc1 = mo.transfer[%tc0] %n1
      : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %t0, %t1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// ============================================================================
// Rejection: a fork unbundle (one element has more than one user) fails
// the single-use precondition.
// ============================================================================

// CHECK-LABEL: mo.graph @reject_fork_unbundle
mo.graph @reject_fork_unbundle(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:1>
  %b = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %ar:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %p:2 = mo.bundled.expand(%arg)
        : <[3], f32, gpu:0>
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%p#0, %p#1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
           !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u:2 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  // %u#0 has two users; the single-use precondition fails and nothing wraps.
  %r0a = mo.relu(%u#0) : !mo.tensor<[3], f32, gpu:0>
  %r0b = mo.relu(%u#0) : !mo.tensor<[3], f32, gpu:0>
  %r1  = mo.relu(%u#1) : !mo.tensor<[3], f32, gpu:1>

  // CHECK: mo.parallel
  // CHECK:   mo.bundled.allreduce.sum
  // CHECK-NOT: mo.parallel
  // CHECK: mo.relu
  // CHECK: mo.relu
  // CHECK: mo.relu

  %t0, %tc0 = mo.transfer[%ar#1] %r0a
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %t0b, %tc0b = mo.transfer[%tc0] %r0b
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %t1, %tc1 = mo.transfer[%tc0b] %r1
      : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %t0, %t0b, %t1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// ============================================================================
// Rejection: a commutative consumer where the unbundle result appears at
// different operand positions on each launch.  No operand position holds
// the unbundle's results in launch-index order, so the anyDriving check
// fires.
// ============================================================================

// CHECK-LABEL: mo.graph @reject_commutative_swap
mo.graph @reject_commutative_swap(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %r0: !mo.tensor<[3], f32, gpu:0>,
    %r1: !mo.tensor<[3], f32, gpu:1>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %c0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:0>
  %c1 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>>
      : !mo.tensor<[3], f32, gpu:1>
  %b = mo.tensor.bundle(%c0, %c1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %ar:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %p:2 = mo.bundled.expand(%arg)
        : <[3], f32, gpu:0>
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
    %out, %ch_out = mo.bundled.allreduce.sum(%p#0, %p#1, %sig0, %sig1, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>,
           !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %u:2 = mo.tensor.unbundle(%ar#0)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  // %s0 has the unbundle result on the LEFT, %s1 on the RIGHT, so no
  // operand position contains the unbundle results in launch order.
  %s0 = mo.add(%u#0, %r0) : !mo.tensor<[3], f32, gpu:0>
  %s1 = mo.add(%r1, %u#1) : !mo.tensor<[3], f32, gpu:1>

  // CHECK: mo.parallel
  // CHECK:   mo.bundled.allreduce.sum
  // CHECK-NOT: mo.parallel
  // CHECK: mo.add
  // CHECK: mo.add

  %t0, %tc0 = mo.transfer[%ar#1] %s0
      : !mo.tensor<[3], f32, gpu:0> to <"cpu", 0>
  %t1, %tc1 = mo.transfer[%tc0] %s1
      : !mo.tensor<[3], f32, gpu:1> to <"cpu", 0>
  mo.output %t0, %t1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}
