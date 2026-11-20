// RUN: mo-opt %s --mo-merge-adjacent-parallels --verify-each | FileCheck %s

// Regression test for the double-erase / use-after-free in
// mo-merge-adjacent-parallels uncovered by FLUX.2 (GEX-3756).
//
// The bug is purely a worklist-management problem: one op can be pushed
// to `toErase` twice when it plays the *consumer* role in one merge
// (during this walk pass) and then the *producer* role in a later
// merge during the same walk pass.  Without dedup, the second
// `.erase()` operates on freed memory; in debug builds ~Operation()'s
// "operation destroyed but still has uses" assertion fires and crashes
// inside the diagnostic printer.

// ============================================================================
// Minimal double-push reproducer (GEX-3756).
//
// Block order: %z, %y, %x (chainless, no buffers).
// SSA dataflow: %z -> %x -> %y (use-before-def of %x in %y, allowed in
// mo.graph).
//
// Walk visits in block order:
//   1. %z: no parallel producer, skip.
//   2. %y: producer = %x. Merge %y into %x -> %merged_xy.  %x's only
//      bundle yield is consumed solely by %y, so dropProducerYield = true
//      and %x.getResult is NOT replaced; %y stays as the lone user of
//      that result.  toErase = [%y, %x].
//   3. %x: walk visits it.  %x.use_empty() is false because %y still
//      references its dropped yield.  findMergeCandidate(%x) succeeds
//      with %z as producer (chainless, no buffer mismatch).  Merge %x
//      into %z -> %merged_zx.  toErase = [%y, %x, %x, %z].
//
// Without dedup, erasing the second %x is a use-after-free; in assert
// builds the destructor's "operation destroyed but still has uses"
// check fires and the diagnostic printer SIGSEGVs on the freed op.
// ============================================================================

// CHECK-LABEL: mo.graph @double_push_reproducer
mo.graph @double_push_reproducer(%arg0: !mo.tensor<[3], f32, gpu:0>,
                                 %arg1: !mo.tensor<[3], f32, gpu:1>)
    -> (!mo.bundle<[3], f32, gpu:0-1>) {
  %b = mo.tensor.bundle(%arg0, %arg1)
      : (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>)
     -> !mo.bundle<[3], f32, gpu:0-1>

  // %z: outermost producer.
  %z = mo.parallel (%arg) in (%b : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }

  // %y appears in the block BEFORE %x but uses %x.  This is legal in
  // mo.graph and is exactly the shape that mergeCandidate produces
  // when it inserts a merged op at the producer slot.
  %y = mo.parallel (%arg) in (%x : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %n = mo.negative(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %n : !mo.tensor<[3], f32, gpu:0>
  }

  // %x consumes %z.  Its sole bundle yield is consumed only by %y
  // above, so the first merge will mark that yield as droppable and
  // leave %x.getResult with %y as its only user -- which keeps %x out
  // of the walk's use_empty() skip.
  %x = mo.parallel (%arg) in (%z : !mo.bundle<[3], f32, gpu:0-1>)
      -> (!mo.bundle<[3], f32, gpu:0-1>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32, gpu:0>
    mo.yield %r : !mo.tensor<[3], f32, gpu:0>
  }

  // Fixed point should collapse all three parallels into one.
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.relu
  // CHECK:       mo.negative
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %y : !mo.bundle<[3], f32, gpu:0-1>
}

// ============================================================================
// Three-layer chain: two allreduces per layer sharing signal buffers,
// followed by a chainless per-device add.  Chains thread across layers.
// All six parallels should collapse into a single merged parallel.
// ============================================================================

// CHECK-LABEL: mo.graph @multi_layer_allreduce_chain
mo.graph @multi_layer_allreduce_chain(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %in_a0: !mo.tensor<[4], f32, gpu:0>, %in_a1: !mo.tensor<[4], f32, gpu:1>,
    %in_b0: !mo.tensor<[4], f32, gpu:0>, %in_b1: !mo.tensor<[4], f32, gpu:1>,
    %in_c0: !mo.tensor<[4], f32, gpu:0>, %in_c1: !mo.tensor<[4], f32, gpu:1>,
    %in_d0: !mo.tensor<[4], f32, gpu:0>, %in_d1: !mo.tensor<[4], f32, gpu:1>,
    %in_e0: !mo.tensor<[4], f32, gpu:0>, %in_e1: !mo.tensor<[4], f32, gpu:1>,
    %in_f0: !mo.tensor<[4], f32, gpu:0>, %in_f1: !mo.tensor<[4], f32, gpu:1>)
    -> (!mo.bundle<[4], f32, gpu:0-1>, !mo.chain) {
  %dt_a = mo.tensor.bundle(%in_a0, %in_a1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %dt_b = mo.tensor.bundle(%in_b0, %in_b1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %dt_c = mo.tensor.bundle(%in_c0, %in_c1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %dt_d = mo.tensor.bundle(%in_d0, %in_d1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %dt_e = mo.tensor.bundle(%in_e0, %in_e1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %dt_f = mo.tensor.bundle(%in_f0, %in_f1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %ch = mo.chain.create()

  // --- Layer 1 ---
  %L1a:2 = mo.parallel (%arg, %bch) in (%dt_a : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L1b:2 = mo.parallel (%arg, %bch) in (%dt_b : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L1a#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L1add = mo.parallel (%arg_a, %arg_b) in (
      %L1a#0 : !mo.bundle<[4], f32, gpu:0-1>,
      %L1b#0 : !mo.bundle<[4], f32, gpu:0-1>)
      -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_b) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %s : !mo.tensor<[4], f32, gpu:0>
  }

  // --- Layer 2 ---
  %L2a:2 = mo.parallel (%arg, %bch) in (%dt_c : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L1b#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L2b:2 = mo.parallel (%arg, %bch) in (%dt_d : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L2a#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L2add = mo.parallel (%arg_a, %arg_b) in (
      %L2a#0 : !mo.bundle<[4], f32, gpu:0-1>,
      %L2b#0 : !mo.bundle<[4], f32, gpu:0-1>)
      -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_b) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %s : !mo.tensor<[4], f32, gpu:0>
  }

  // --- Layer 3 ---
  %L3a:2 = mo.parallel (%arg, %bch) in (%dt_e : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L2b#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L3b:2 = mo.parallel (%arg, %bch) in (%dt_f : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L3a#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L3add = mo.parallel (%arg_a, %arg_b) in (
      %L3a#0 : !mo.bundle<[4], f32, gpu:0-1>,
      %L3b#0 : !mo.bundle<[4], f32, gpu:0-1>)
      -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_b) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %s : !mo.tensor<[4], f32, gpu:0>
  }

  // After the fixed-point merge, each layer's three parallels (two
  // chained allreduces + one chainless add) should each collapse into
  // a single merged parallel.  The three layers remain separate because
  // they have no direct parallel-to-parallel SSA input dependency (only
  // chain dependencies).
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.relu
  // CHECK:       mo.add
  // CHECK:       mo.yield
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.relu
  // CHECK:       mo.add
  // CHECK:       mo.yield
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.relu
  // CHECK:       mo.add
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %L3add, %L3b#1
      : !mo.bundle<[4], f32, gpu:0-1>, !mo.chain
}

// ============================================================================
// Cross-layer cascade: the residual-add output from layer 1 feeds as an
// input to layer 2.  This creates a direct parallel-to-parallel SSA
// dependency across layers, exercising the path where a merged op is
// immediately consumed by the next layer's merge.
// ============================================================================

// CHECK-LABEL: mo.graph @cross_layer_cascade
mo.graph @cross_layer_cascade(
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %in_a0: !mo.tensor<[4], f32, gpu:0>, %in_a1: !mo.tensor<[4], f32, gpu:1>,
    %in_b0: !mo.tensor<[4], f32, gpu:0>, %in_b1: !mo.tensor<[4], f32, gpu:1>,
    %in_c0: !mo.tensor<[4], f32, gpu:0>, %in_c1: !mo.tensor<[4], f32, gpu:1>)
    -> (!mo.bundle<[4], f32, gpu:0-1>, !mo.chain) {
  %dt_a = mo.tensor.bundle(%in_a0, %in_a1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %dt_b = mo.tensor.bundle(%in_b0, %in_b1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %dt_c = mo.tensor.bundle(%in_c0, %in_c1)
      : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>)
     -> !mo.bundle<[4], f32, gpu:0-1>
  %ch = mo.chain.create()

  // Layer 1: allreduce pair + add
  %L1a:2 = mo.parallel (%arg, %bch) in (%dt_a : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%ch) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L1b:2 = mo.parallel (%arg, %bch) in (%dt_b : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L1a#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L1add = mo.parallel (%arg_a, %arg_b) in (
      %L1a#0 : !mo.bundle<[4], f32, gpu:0-1>,
      %L1b#0 : !mo.bundle<[4], f32, gpu:0-1>)
      -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %s = mo.add(%arg_a, %arg_b) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %s : !mo.tensor<[4], f32, gpu:0>
  }

  // Layer 2: a single allreduce that takes layer 1's add output as
  // input, followed by a per-device relu.  This creates a direct
  // SSA chain: L1add result -> L2a input -> L2relu input.
  %L2a:2 = mo.parallel (%arg, %bch) in (%L1add : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L1b#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }
  %L2relu = mo.parallel (%arg) in (
      %L2a#0 : !mo.bundle<[4], f32, gpu:0-1>)
      -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %r = mo.relu(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %r : !mo.tensor<[4], f32, gpu:0>
  }

  // Layer 3: another allreduce consuming the relu from layer 2.
  %L3a:2 = mo.parallel (%arg, %bch) in (%L2relu : !mo.bundle<[4], f32, gpu:0-1>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>,
              %sig1 : !mo.buffer<[1], ui8, gpu:1>)
      chain(%L2a#1) -> (!mo.bundle<[4], f32, gpu:0-1>) {
    %x = mo.negative(%arg) : !mo.tensor<[4], f32, gpu:0>
    mo.yield %x, %bch : !mo.tensor<[4], f32, gpu:0>, !mo.chain
  }

  // With cross-layer SSA dependencies, the fixed-point merge should
  // cascade everything into a single parallel.
  // CHECK:     mo.parallel
  // CHECK:       mo.relu
  // CHECK:       mo.relu
  // CHECK:       mo.add
  // CHECK:       mo.relu
  // CHECK:       mo.relu
  // CHECK:       mo.negative
  // CHECK:       mo.yield
  // CHECK-NOT: mo.parallel

  mo.output %L3a#0, %L3a#1
      : !mo.bundle<[4], f32, gpu:0-1>, !mo.chain
}
