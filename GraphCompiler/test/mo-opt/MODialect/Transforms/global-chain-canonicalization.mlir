// RUN: mo-opt --mo-canonicalize-chain-sequences %s | FileCheck %s --enable-var-scope

!tt = !mo.tensor<[3, 3], f32>
!bt = !mo.buffer<[3, 3], f32>
!ct = !mo.chain
!ot = !mo.opaque<"T">

// CHECK-LABEL: mo.graph @one_long_sequence
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: {{%.*}}, [[ch1:%.*]] = mo.mutable.load({{.*}}, [[ch0]]
// CHECK: mo.mutable.store({{.*}}, [[ch1]]
// CHECK: {{%.*}}, [[ch2:%.*]] = mo.mutable.load({{.*}}, [[ch0]]
// CHECK: mo.mutable.store({{.*}}, [[ch2]]
mo.graph @one_long_sequence(%buf0: !bt, %buf1: !bt, %rhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create()

  %vbuf, %ch1 = mo.mutable.load(%buf0, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %res1 = mo.add(%rhs, %vbuf) : !tt
  %ch2 = mo.mutable.store(%buf0, %res1, %ch1) : (!bt, !tt, !ct) -> (!ct)

  %vbuf2, %ch3 = mo.mutable.load(%buf1, %ch2) : (!bt, !ct) -> (!tt, !ct)
  %res2 = mo.negative(%vbuf2) : !tt
  %ch4 = mo.mutable.store(%buf1, %res2, %ch3) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf0, %buf1 : !bt, !bt
}

// CHECK-LABEL: mo.graph @interleaved_sequence
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create
// CHECK-DAG: {{%.*}}, [[ch1:%.*]] = mo.mutable.load({{%.*}}, [[ch0]]
// CHECK-DAG: mo.mutable.store({{.*}}, [[ch1]]
// CHECK-DAG: {{%.*}}, [[ch2:%.*]] = mo.mutable.load({{%.*}}, [[ch0]]
// CHECK-DAG: mo.mutable.store({{.*}}, [[ch2]]
mo.graph @interleaved_sequence(%buf0: !bt, %buf1: !bt, %lhs: !tt) -> (!bt, !bt)
{
  %ch0 = mo.chain.create()

  %vbuf, %ch1 = mo.mutable.load(%buf0, %ch0) : (!bt, !ct) -> (!tt, !ct)
  %vbuf2, %ch2 = mo.mutable.load(%buf1, %ch1) : (!bt, !ct) -> (!tt, !ct)

  %res1 = mo.add(%lhs, %vbuf) : !tt
  %res2 = mo.negative(%vbuf2) : !tt

  %ch3 = mo.mutable.store(%buf0, %res1, %ch2) : (!bt, !tt, !ct) -> (!ct)
  %ch4 = mo.mutable.store(%buf1, %res2, %ch3) : (!bt, !tt, !ct) -> (!ct)

  mo.output %buf0, %buf1 : !bt, !bt
}

// CHECK-LABEL: mo.graph @custom_ops
// CHECK-SAME: [[buf0:%.*]]:{{.*}}[[buf1:%.*]]:
// CHECK-DAG: [[ch0:%.*]] = mo.chain.create
// CHECK-DAG: [[ch1:%.*]] = mo.custom [symbol = "foo"]([[buf0]], [[ch0]])
// CHECK-DAG: [[ch2:%.*]] = mo.custom [symbol = "foo"]([[buf1]], [[ch0]])
mo.graph @custom_ops(%buf0: !bt, %buf1: !bt)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom [symbol = "foo", device = #M.device_ref<"cpu", 0>](%buf0, %ch0) : (!bt, !ct) -> (!ct)
  %ch2 = mo.custom [symbol = "foo", device = #M.device_ref<"cpu", 0>](%buf1, %ch1) : (!bt, !ct) -> (!ct)

  mo.output
}

// CHECK-LABEL: mo.graph @custom_ops_opaque
// CHECK-SAME: [[BUF:%.*]]:{{.*}}[[OPAQUE:%.*]]:
mo.graph @custom_ops_opaque(%buf: !bt, %opaque: !ot)
{
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // Pair of custom ops whose linear chains should get decomposed.
  // CHECK-DAG: [[CH1:%.*]] = mo.custom{{.*}}foo{{.*}}([[BUF]], [[CH0]])
  // CHECK-DAG: [[CH2:%.*]] = mo.custom{{.*}}bar{{.*}}([[OPAQUE]], [[CH0]])
  %ch1 = mo.custom [symbol = "foo", device = #M.device_ref<"cpu", 0>](%buf, %ch0) : (!bt, !ct) -> (!ct)
  %ch2 = mo.custom [symbol = "bar", device = #M.device_ref<"cpu", 0>] (%opaque, %ch1) : (!ot, !ct) -> (!ct)

  // Check that this true linear dependency on %opaque remains.
  // CHECK-DAG: [[CH3:%.*]] = mo.custom{{.*}}baz{{.*}}([[OPAQUE]], [[CH2]])
  %ch3 = mo.custom [symbol = "baz", device = #M.device_ref<"cpu", 0>] (%opaque, %ch2) : (!ot, !ct) -> (!ct)

  mo.output
}

// CHECK-LABEL: mo.graph @different_opaque_vals_same_type
// CHECK-SAME: [[OPAQUE1:%.*]]: !mo.opaque<"T">, [[OPAQUE2:%.*]]: !mo.opaque<"T">
mo.graph @different_opaque_vals_same_type(%opaque1: !ot, %opaque2: !ot)
{
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // Check that we don't move baz past bar even though since they operate on an
  // opaque value of the same type (and we do have the potential of aliasing).

  // CHECK-DAG: [[CH1:%.*]] = mo.custom{{.*}}bar{{.*}}([[OPAQUE1]], [[CH0]])
  // CHECK-DAG: [[CH2:%.*]] = mo.custom{{.*}}baz{{.*}}([[OPAQUE2]], [[CH1]])
  %ch1 = mo.custom [symbol = "bar", device = #M.device_ref<"cpu", 0>] (%opaque1, %ch0) : (!ot, !ct) -> (!ct)
  %ch2 = mo.custom [symbol = "baz", device = #M.device_ref<"cpu", 0>] (%opaque2, %ch1) : (!ot, !ct) -> (!ct)

  mo.output
}

// CHECK-LABEL: @everything
// CHECK-SAME: ([[arg0:%.*]]: !mo.buffer<[3, 3], f32>, [[arg1:%.*]]: !mo.buffer<[3, 3], f32>, [[arg2:%.*]]: !mo.tensor<[3, 3], f32>) {
// CHECK-DAG:  [[a0:%.*]] = mo.chain.create()
// CHECK-DAG:  [[a1:%.*]] = mo.custom [symbol = "foo"]([[arg0]], [[a0]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> !mo.chain

// CHECK-DAG:  [[outTensor_1:%.*]], [[outChain_0:%.*]] = mo.mutable.load([[arg0]], [[a1]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> (!mo.tensor<[3, 3], f32>, !mo.chain)
// CHECK-DAG:  [[a3:%.*]] = mo.add([[outTensor_1]], [[arg2]]) : !mo.tensor<[3, 3], f32>
// CHECK-DAG:  [[ch0:%.*]] = mo.chain.create([[outChain_0]], [[a1]])
// CHECK-DAG:  [[a5:%.*]] = mo.mutable.store([[arg0]], [[a3]], [[ch0]]) : (!mo.buffer<[3, 3], f32>, !mo.tensor<[3, 3], f32>, !mo.chain) -> !mo.chain

// CHECK-DAG:  [[outTensor:%.*]], [[outChain:%.*]] = mo.mutable.load([[arg1]], [[a0]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> (!mo.tensor<[3, 3], f32>, !mo.chain)
// CHECK-DAG:  [[a2:%.*]] = mo.add([[outTensor]], [[arg2]]) : !mo.tensor<[3, 3], f32>
// CHECK-DAG:  [[a4:%.*]] = mo.mutable.store([[arg1]], [[a2]], [[outChain]]) : (!mo.buffer<[3, 3], f32>, !mo.tensor<[3, 3], f32>, !mo.chain) -> !mo.chain

// CHECK-DAG:  [[ch2:%.*]] = mo.chain.create([[a4]], [[outChain]])
// CHECK-DAG:  [[a6:%.*]] = mo.custom [symbol = "foo"]([[arg1]], [[ch2]]) : (!mo.buffer<[3, 3], f32>, !mo.chain) -> !mo.chain

// CHECK: mo.output
mo.graph @everything(%buf0: !bt, %buf1: !bt, %tensor: !tt)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom [symbol = "foo", device = #M.device_ref<"cpu", 0>](%buf0, %ch0) : (!bt, !ct) -> (!ct)

  %vbuf, %ch2 = mo.mutable.load(%buf1, %ch1) : (!bt, !ct) -> (!tt, !ct)
  %vbuf2, %ch3 = mo.mutable.load(%buf0, %ch2) : (!bt, !ct) -> (!tt, !ct)

  %res1 = mo.add(%vbuf, %tensor) : !tt
  %res2 = mo.add(%vbuf2, %tensor) : !tt

  %ch4 = mo.mutable.store(%buf1, %res1, %ch3) : (!bt, !tt, !ct) -> (!ct)
  %ch5 = mo.mutable.store(%buf0, %res2, %ch4) : (!bt, !tt, !ct) -> (!ct)

  %ch6 = mo.custom [symbol = "foo", device = #M.device_ref<"cpu", 0>](%buf1, %ch5) : (!bt, !ct) -> (!ct)

  mo.output
}

// Tests that chain canonicalization works for single-chain call ops.
// Note: the `mo.call` and `mo.mutable.load` can proceed in parallel.
// CHECK-LABEL: mo.graph @call_single_chain
// CHECK: [[CHAIN:%.*]] = mo.chain.create
// CHECK: [[BUF2:%.*]] = mo.buffer.create
// CHECK: [[CALL_CHAIN:%.*]] = mo.call @store_buffer([[BUF2]], %arg1, [[CHAIN]])
// CHECK: [[LOAD:%.*]], [[LOAD_CHAIN:%.*]] = mo.mutable.load(%arg0, [[CHAIN]])
// CHECK: mo.mutable.store([[BUF2]], [[LOAD]], [[CALL_CHAIN]])
mo.graph @call_single_chain(%buf: !bt, %tensor: !tt)
{
  %ch0 = mo.chain.create()
  %buf2 = mo.buffer.create : !bt
  %ch1 = mo.call @store_buffer(%buf2, %tensor, %ch0) : (!bt, !tt, !ct) -> (!ct)
  %loaded, %ch2 = mo.mutable.load(%buf, %ch1) : (!bt, !ct) -> (!tt, !ct)
  %ch3 = mo.mutable.store(%buf2, %loaded, %ch2) : (!bt, !tt, !ct) -> (!ct)
  mo.output
}

// Tests that chain canonicalization bails out for multi-chain call ops.
// CHECK-LABEL: mo.graph @call_multi_chain
// CHECK: [[BUF2:%.*]] = mo.buffer.create
// CHECK: [[CALL:%.*]]:2 = mo.call @multi_chain_callee([[BUF2]], %arg1, %arg2, %arg3)
// CHECK: mo.mutable.load(%arg0, [[CALL]]#0
// CHECK: mo.output
mo.graph @call_multi_chain(%buf: !bt, %tensor: !tt, %main: !ct, %dev: !ct)
{
  %buf2 = mo.buffer.create : !bt
  %main_out, %dev_out = mo.call @multi_chain_callee(%buf2, %tensor, %main, %dev)
      : (!bt, !tt, !ct, !ct) -> (!ct, !ct)
  %loaded, %ch = mo.mutable.load(%buf, %main_out) : (!bt, !ct) -> (!tt, !ct)
  %ch_store = mo.mutable.store(%buf2, %loaded, %ch) : (!bt, !tt, !ct) -> (!ct)
  mo.output
}

mo.graph @store_buffer(%buf: !bt, %tensor: !tt, %ch_in: !ct)
    -> (!bt, !ct) isSubgraph
{
  %loaded, %ch1 = mo.mutable.load(%buf, %ch_in)
      : (!bt, !ct) -> (!tt, !ct)
  %updated = mo.add(%loaded, %tensor) : !tt
  %ch2 = mo.mutable.store(%buf, %updated, %ch1)
      : (!bt, !tt, !ct) -> (!ct)
  mo.output %buf, %ch2 : !bt, !ct
}

mo.graph @multi_chain_callee(%buf: !bt, %tensor: !tt, %main: !ct, %dev: !ct)
    -> (!bt, !ct, !ct) isSubgraph
{
  mo.output %buf, %main, %dev : !bt, !ct, !ct
}

// CHECK-LABEL: mo.graph @chain_canonicalization_bug
// CHECK: (%[[BUF0:.*]]: !mo.buffer<[3, 3], f32>, %[[BUF1:.*]]: !mo.buffer<[3, 3], f32>, %[[TENSOR0:.*]]: !mo.tensor<[3, 3], f32>, %[[TENSOR1:.*]]: !mo.tensor<[3, 3], f32>, %[[INIT0:.*]]: !mo.chain, %[[INIT1:.*]]: !mo.chain)
mo.graph @chain_canonicalization_bug(
  %buf0: !bt,
  %buf1: !bt,
  %tensor0: !tt,
  %tensor1: !tt,
  %init0: !ct,
  %init1: !ct
) -> (!tt, !tt, !ct, !ct) {
  // COM: Store to %buf0 which is merged with another chain. The chain.create
  // COM: operation is important to trigger the correct rewrite pattern.

  // CHECK-DAG: %[[CH1:.*]] = mo.mutable.store(%[[BUF0]], %[[TENSOR0]], %[[INIT0]])
  %ch1 = mo.mutable.store(%buf0, %tensor0, %init0) : (!bt, !tt, !ct) -> !ct

  // COM: Chain create to generate a composite chain
  %merged = mo.chain.create(%ch1, %init0) : !ct, !ct

  // Independent loads from buf0 and buf1 sequenced after the store.
  // These will be split up such that only the load from %buf0 is sequenced
  // after the store operation.

  // CHECK-DAG: %[[MERGE1:.*]] = mo.chain.create(%[[CH1]], %[[INIT0]])
  // CHECK-DAG: %[[OUT_TENSOR0:.*]], %[[OUT_CHAIN0:.*]] = mo.mutable.load(%[[BUF1]], %[[INIT0]])
  %vbuf1, %ch6 = mo.mutable.load(%buf1, %merged) : (!bt, !ct) -> (!tt, !ct)

  // CHECK-DAG: %[[MERGE2:.*]] = mo.chain.create(%[[CH1]], %[[OUT_CHAIN0]]) : !mo.chain, !mo.chain
  // CHECK-DAG: %[[OUT_TENSOR1:.*]], %[[OUT_CHAIN1:.*]] = mo.mutable.load(%[[BUF0]], %[[MERGE1]])
  %vbuf2, %ch7 = mo.mutable.load(%buf0, %merged) : (!bt, !ct) -> (!tt, !ct)

  // COM: Ensure the returned chains only sequence on the correct buffers.

  // CHECK: mo.output %[[OUT_TENSOR0]], %[[OUT_TENSOR1]], %[[MERGE2]], %[[OUT_CHAIN0]]
  mo.output %vbuf1, %vbuf2, %ch6, %ch7 : !tt, !tt, !ct, !ct
}

// A collective is a rendezvous: relaxing its chain against unrelated
// same-device work is race-free but can strand the other ranks. Neither
// direction of the pattern may reorder across one.

!sb0 = !mo.buffer<[16], ui8, gpu:0>
!sb1 = !mo.buffer<[16], ui8, gpu:1>
!kv0 = !mo.buffer<[8, 8], f32, gpu:0>
!kv1 = !mo.buffer<[8, 8], f32, gpu:1>
!gt0 = !mo.tensor<[4], ui32, gpu:0>
!gt1 = !mo.tensor<[4], ui32, gpu:1>

// CHECK-LABEL: mo.graph @collective_then_custom
mo.graph @collective_then_custom(%off: !gt1, %s0: !sb0, %s1: !sb1, %kv0: !kv0)
    -> !ct {
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // The custom writes only %kv0 and the broadcast only the signal buffers, so
  // the aliasing check alone would allow the reorder.
  // CHECK: [[OUT:%.*]]:2, [[BCH:%.*]] = mo.distributed.broadcast({{.*}}, [[CH0]])
  %o:2, %bch = mo.distributed.broadcast(%off, %s0, %s1, %ch0) [root = 1]
      : (!gt1, !sb0, !sb1, !ct) -> (!gt0, !gt1, !ct)

  // CHECK: mo.custom{{.*}}matmul{{.*}}({{.*}}, [[BCH]])
  %mch = mo.custom [symbol = "matmul", device = #M.device_ref<"gpu", 0>](%kv0, %bch)
      : (!kv0, !ct) -> (!ct)

  mo.output %mch : !ct
}

// The same, with the mo.chain.create the graph builder emits ahead of a
// collective when the per-device chains have diverged. This exercises the
// pattern's chain.create branch, which must not hoist operands out of the
// merge either.
// CHECK-LABEL: mo.graph @collective_through_merge
mo.graph @collective_through_merge(%off: !gt1, %s0: !sb0, %s1: !sb1,
                                   %kv0: !kv0, %kv1: !kv1) -> !ct {
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // CHECK: [[PRE:%.*]] = mo.custom{{.*}}prev{{.*}}({{.*}}, [[CH0]])
  %pre = mo.custom [symbol = "prev", device = #M.device_ref<"gpu", 1>](%kv1, %ch0)
      : (!kv1, !ct) -> (!ct)
  %merged = mo.chain.create(%ch0, %pre) : !ct, !ct

  // CHECK: [[OUT:%.*]]:2, [[BCH:%.*]] = mo.distributed.broadcast({{.*}}, [[PRE]])
  %o:2, %bch = mo.distributed.broadcast(%off, %s0, %s1, %merged) [root = 1]
      : (!gt1, !sb0, !sb1, !ct) -> (!gt0, !gt1, !ct)

  // CHECK: mo.custom{{.*}}matmul{{.*}}({{.*}}, [[BCH]])
  %mch = mo.custom [symbol = "matmul", device = #M.device_ref<"gpu", 0>](%kv0, %bch)
      : (!kv0, !ct) -> (!ct)

  mo.output %mch : !ct
}

// Without a collective in the sequence the same shape still relaxes.
// CHECK-LABEL: mo.graph @no_collective_still_relaxes
mo.graph @no_collective_still_relaxes(%kv0: !kv0, %kv1: !kv1) -> !ct {
  // CHECK: [[CH0:%.*]] = mo.chain.create()
  %ch0 = mo.chain.create()

  // CHECK-DAG: [[A:%.*]] = mo.custom{{.*}}a{{.*}}({{.*}}, [[CH0]])
  %c1 = mo.custom [symbol = "a", device = #M.device_ref<"gpu", 0>](%kv0, %ch0)
      : (!kv0, !ct) -> (!ct)

  // CHECK-DAG: [[B:%.*]] = mo.custom{{.*}}b{{.*}}({{.*}}, [[CH0]])
  %c2 = mo.custom [symbol = "b", device = #M.device_ref<"gpu", 1>](%kv1, %c1)
      : (!kv1, !ct) -> (!ct)

  // CHECK: mo.chain.create([[B]], [[A]])
  mo.output %c2 : !ct
}
