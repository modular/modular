// RUN: mo-opt %s --mo-merge-adjacent-parallels --verify-each | FileCheck %s

// Unit test for the merge-adjacent-parallels pass in isolation.  Tests the
// full create -> canonicalize -> merge pipeline live in
// create-parallel-from-unbundle.mlir and the integration suite.

// ============================================================================
// Simple merge: B consumes A's single bundle output directly (no buffers,
// no chain).  Merged op has the concatenated body and yields both values.
// ============================================================================

// CHECK-LABEL: mo.graph @merge_simple
mo.graph @merge_simple(%arg0: !mo.tensor<[3], f32, gpu:0>,
                       %arg1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %a = mo.parallel (%arg) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }
  %c = mo.parallel (%arg) in (%a
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }

  // Exactly one merged parallel with both ops in sequence.  B's block arg
  // is replaced by A's yield operand (%r), and A's yield is dropped since
  // it had no external users beyond the now-absorbed B.
  // CHECK:     mo.parallel
  // CHECK:       %[[R:[a-zA-Z0-9_]+]] = mo.relu
  // CHECK:       %[[N:[a-zA-Z0-9_]+]] = mo.negative(%[[R]])
  // CHECK:       mo.yield %[[N]] :
  // CHECK-NOT: mo.parallel

  mo.output %c
      : !mo.bundle<[3], f32, gpu:0-1>
}

// ============================================================================
// Non-shared consumer input bundle is defined between producer and consumer.
// The pass does not hoist it: the merged op is inserted at the producer slot
// and may reference that bundle before its defining op (allowed in mo.graph).
// ============================================================================

// CHECK-LABEL: mo.graph @merge_with_non_shared_between
mo.graph @merge_with_non_shared_between(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %r0: !mo.tensor<[3], f32, gpu:0>,   %r1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %a = mo.parallel (%arg) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x : !mo.tensor<[3], f32, gpu:0>
  }
  // This bundle is constructed AFTER %a but before %c; merged parallel is
  // still inserted before %a and takes it as an operand.
  %residuals = mo.tensor.bundle(%r0, %r1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %c = mo.parallel (%arg_a, %arg_r) in (%a
        : !mo.bundle<[3], f32, gpu:0-1>,
                                           %residuals
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_r) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %s : !mo.tensor<[3], f32, gpu:0>
  }

  // One merged parallel; residuals bundle may appear after it in the block.
  // CHECK:     mo.tensor.bundle
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.add
  // CHECK:     mo.tensor.bundle
  // CHECK-NOT: mo.parallel

  mo.output %c
      : !mo.bundle<[3], f32, gpu:0-1>
}

// ============================================================================
// Buffers + chain on BOTH sides: merge requires identical buffer lists and
// strict chain threading (B.inChain == A.outChain, no other users).
// ============================================================================

// CHECK-LABEL: mo.graph @merge_chain_both
mo.graph @merge_chain_both(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>, !mo.chain) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %a:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %c:2 = mo.parallel (%arg, %bch) in (%a#0
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%a#1)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n, %bch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }

  // One merged parallel that keeps A's buffers + inChain; outChain is a
  // single trailing result.
  // CHECK: %[[M:[a-zA-Z0-9_]+]]:{{[0-9]+}} = mo.parallel
  // CHECK-SAME: buffers(%{{.*}} : !mo.buffer<[1], ui8, gpu:0>, %{{.*}} : !mo.buffer<[1], ui8, gpu:1>)
  // CHECK:   mo.relu
  // CHECK:   mo.negative
  // CHECK:   mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %c#0, %c#1
      : !mo.bundle<[3], f32, gpu:0-1>,
        !mo.chain
}

// ============================================================================
// Buffer mismatch on BOTH sides: merge must be rejected, two parallels
// remain.
// ============================================================================

// CHECK-LABEL: mo.graph @no_merge_buffer_mismatch
mo.graph @no_merge_buffer_mismatch(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %sigA0: !mo.buffer<[1], ui8, gpu:0>, %sigA1: !mo.buffer<[1], ui8, gpu:1>,
    %sigB0: !mo.buffer<[1], ui8, gpu:0>, %sigB1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>, !mo.chain) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %a:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sigA0 : !mo.buffer<[1], ui8, gpu:0>,
              %sigA1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %c:2 = mo.parallel (%arg, %bch) in (%a#0
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sigB0 : !mo.buffer<[1], ui8, gpu:0>,
              %sigB1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%a#1)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n, %bch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }

  // Different buffers on each side — the merge must bail; both parallels
  // remain.
  // CHECK: mo.parallel
  // CHECK: mo.parallel

  mo.output %c#0, %c#1
      : !mo.bundle<[3], f32, gpu:0-1>,
        !mo.chain
}

// ============================================================================
// One-sided buffers + chain: producer has buffers+chain, consumer doesn't.
// Merged op inherits producer's buffers+chain.
// ============================================================================

// CHECK-LABEL: mo.graph @merge_chain_producer_only
mo.graph @merge_chain_producer_only(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>, !mo.chain) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %ch = mo.chain.create()
  %a:2 = mo.parallel (%arg, %bch) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %c = mo.parallel (%arg) in (%a#0
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }

  // Merged op carries A's buffers and chain.
  // CHECK:     mo.parallel
  // CHECK-SAME: buffers(%{{.*}} : !mo.buffer<[1], ui8, gpu:0>, %{{.*}} : !mo.buffer<[1], ui8, gpu:1>)
  // CHECK:       mo.relu
  // CHECK:       mo.negative
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %c, %a#1
      : !mo.bundle<[3], f32, gpu:0-1>,
        !mo.chain
}

// ============================================================================
// Multi-producer chain: consumer C fans in from A and B, but B itself takes
// A's output as input.  Picking A as the producer would put B's output (a
// non-shared input to C) inside a cycle through the merged op, so the pass
// must pick B.  A subsequent iteration folds A into the merged BC.
// ============================================================================

// CHECK-LABEL: mo.graph @merge_multi_producer_chain
mo.graph @merge_multi_producer_chain(%arg0: !mo.tensor<[3], f32, gpu:0>,
                                     %arg1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %a = mo.parallel (%arg) in (%b
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }
  %b2 = mo.parallel (%arg) in (%a
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }
  %c = mo.parallel (%arg_a, %arg_b) in (%a
        : !mo.bundle<[3], f32, gpu:0-1>,
                                          %b2
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_b) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %s : !mo.tensor<[3], f32, gpu:0>
  }

  // Two iterations should collapse all three into a single parallel: B
  // absorbs C first (picking A as the producer fails the dependence
  // check because B's output transitively depends on A); then A absorbs
  // the merged BC.
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.negative
  // CHECK:       mo.add
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %c
      : !mo.bundle<[3], f32, gpu:0-1>
}

// ============================================================================
// Multi-producer siblings: A and B are independent, both feed consumer C.
// First iteration picks A (first in input order); the resulting merged AC
// still has B as a parallel producer, and the second iteration folds B in.
// Final state: a single parallel with all three bodies.
// ============================================================================

// CHECK-LABEL: mo.graph @merge_multi_producer_independent
mo.graph @merge_multi_producer_independent(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %brg0: !mo.tensor<[3], f32, gpu:0>, %brg1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %ab = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %bb = mo.tensor.bundle(%brg0, %brg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %a = mo.parallel (%arg) in (%ab
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }
  %b3 = mo.parallel (%arg) in (%bb
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }
  %c = mo.parallel (%arg_a, %arg_b) in (%a
        : !mo.bundle<[3], f32, gpu:0-1>,
                                          %b3
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_b) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %s : !mo.tensor<[3], f32, gpu:0>
  }

  // All three parallels collapse into one.  Order inside the merged
  // body: B's compute, then A's compute, then C's add (B is the
  // outermost producer after both iterations because the iteration-2
  // merge places the result at B's slot).
  // CHECK:     mo.parallel
  // CHECK:       mo.negative
  // CHECK:       mo.relu
  // CHECK:       mo.add
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %c
      : !mo.bundle<[3], f32, gpu:0-1>
}

// ============================================================================
// Multi-producer cycle: consumer C has inputs from A, B, AND an external
// per-device value X that itself transitively depends on both A and B.
// Picking either A or B as the producer would put X in a position to
// cycle through the merged op, so both choices fail the dependence check
// and the merge is rejected.  All three parallels (A, B, C) must remain.
// ============================================================================

// CHECK-LABEL: mo.graph @no_merge_multi_producer_all_cycle
mo.graph @no_merge_multi_producer_all_cycle(
    %arg0: !mo.tensor<[3], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %brg0: !mo.tensor<[3], f32, gpu:0>, %brg1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %ab = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %bb = mo.tensor.bundle(%brg0, %brg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %a = mo.parallel (%arg) in (%ab
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }
  %b4 = mo.parallel (%arg) in (%bb
        : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }
  // Per-device adds that fan in from BOTH A and B, then rebundled into the
  // external value X.  Picking A as C's producer fails because X depends
  // on A; picking B fails because X depends on B.
  %a0, %a1 = mo.tensor.unbundle(%a)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %bb0, %bb1 = mo.tensor.unbundle(%b4)
      : (!mo.bundle<[3], f32, gpu:0-1>)
     -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
  %x0 = mo.add(%a0, %bb0) : !mo.tensor<[3], f32, gpu:0>
  %x1 = mo.add(%a1, %bb1) : !mo.tensor<[3], f32, gpu:1>
  %x = mo.tensor.bundle(%x0, %x1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>
  %c = mo.parallel (%arg_a, %arg_b, %arg_x) in (%a
        : !mo.bundle<[3], f32, gpu:0-1>,
                                                  %b4
        : !mo.bundle<[3], f32, gpu:0-1>,
                                                  %x
        : !mo.bundle<[3], f32, gpu:0-1>) -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_x) : !mo.tensor<[3], f32, gpu:0>
    %t = mo.add(%s, %arg_b) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %t : !mo.tensor<[3], f32, gpu:0>
  }

  // All three parallels must remain unmerged.
  // CHECK:     mo.parallel
  // CHECK:     mo.parallel
  // CHECK:     mo.parallel

  mo.output %c
      : !mo.bundle<[3], f32, gpu:0-1>
}
