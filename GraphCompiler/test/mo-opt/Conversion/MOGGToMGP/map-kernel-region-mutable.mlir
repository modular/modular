// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-mutable-loads --map-outline-structured-kernel-bodies --map-to-mogg --mogg-conform-call-operands --mogg-outline-routines --mogg-to-mgp > %t
// RUN: FileCheck %s < %t

// A `!mo.buffer` operand of a new-path mogg.routine is written in place by the
// body, so the region must give it the mutable signature (`m`) and not the
// input one (`i`): MGP reads that signature to decide whether the kernel writes
// the buffer, so an input-signature buffer makes it treat the write as a read
// and the store is lost between executions.
// CHECK-LABEL: mgp.model @mutable_store
// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%{{.*}}) : (%{{.*}}: !mef.chain,
// CHECK: inOutSignature = <"iim.">
mo.graph @mutable_store(%arg0: !mo.tensor<[1], si32>) -> !mo.tensor<[1], si32> {
  %0 = mo.chain.create()
  %1 = mo.buffer.create [initValue = #M.dense_array<0> : tensor<1xsi32>] : !mo.buffer<[1], si32>
  %outTensor, %outChain = mo.mutable.load(%1, %0) : (!mo.buffer<[1], si32>, !mo.chain) -> (!mo.tensor<[1], si32>, !mo.chain)
  %2 = mo.mutable.store(%1, %arg0, %outChain) : (!mo.buffer<[1], si32>, !mo.tensor<[1], si32>, !mo.chain) -> !mo.chain
  mo.output %outTensor : !mo.tensor<[1], si32>
}

// Two kernels writing one buffer stay ordered: the second region waits on the
// first one's chain. The region takes its in-chain from the op's own `$chain`,
// so dropping that edge for the ambient execute chain would leave the two
// writes ordered only by however their chains happen to be joined.
// CHECK-LABEL: mgp.model @two_writes_chained
// CHECK: %[[STORE1:[0-9]+]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%{{[0-9]+}},
// CHECK: %[[CHAIN2:[0-9]+]] = grt.chain.create (%arg0, %[[STORE1]])
// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CHAIN2]],
mo.graph @two_writes_chained(%a: !mo.tensor<[4], si32>, %b: !mo.tensor<[4], si32>) -> (!mo.buffer<[4], si32>, !mo.chain) {
  %ch0 = mo.chain.create()
  %buf = mo.buffer.create : !mo.buffer<[4], si32>
  %ch1 = mo.mutable.store(%buf, %a, %ch0) : (!mo.buffer<[4], si32>, !mo.tensor<[4], si32>, !mo.chain) -> !mo.chain
  %ch2 = mo.mutable.store(%buf, %b, %ch1) : (!mo.buffer<[4], si32>, !mo.tensor<[4], si32>, !mo.chain) -> !mo.chain
  mo.output %buf, %ch2 : !mo.buffer<[4], si32>, !mo.chain
}
