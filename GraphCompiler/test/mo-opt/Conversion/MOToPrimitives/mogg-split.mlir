// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --canonicalize --cse | FileCheck %s

// CHECK-LABEL: mgp.model @mogg_split : mof
// CHECK: } execute (
// CHECK-SAME:   %[[CH:.*]]: !mef.chain,
// CHECK-SAME:   %[[CTX:.*]]: !mgp.context,
// CHECK-SAME:   %[[INPUT:[a-z0-9]*]]: !mgp.tensor<?x8xf32>,
// CHECK-SAME:   %[[DUMMY:[a-z0-9]*]]: !mgp.tensor<?x?xsi64>
// CHECK-SAME: ) -> (!mef.chain, !mgp.tensor<?x8xf32>, !mgp.tensor<?x8xf32>) {
// CHECK-NEXT:   %[[THIRTYTWO:.*]] = index.constant 32
// CHECK-NEXT:   %[[ZERO:.*]] = index.constant 0

// CHECK-NEXT:   %[[DUMMY_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[DUMMY]]
// CHECK-NEXT:   %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %[[DUMMY_SPEC]]
// CHECK-NEXT:   %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %[[DUMMY_SPEC]]
// CHECK-NEXT:   %[[CASTED0:.*]] = index.casts %[[DIM0]] : index to si64
// CHECK-NEXT:   %[[CASTED1:.*]] = index.casts %[[DIM1]] : index to si64
// CHECK-NEXT:   %[[INPUT_BUF:.*]] = mgp.tensor.extract.buffer %[[INPUT]]
// CHECK-NEXT:   %[[INDEX0:.*]] = index.casts %[[CASTED0]] : si64 to index
// CHECK-NEXT:   %[[SPLIT_SPEC1:.*]] = mgp.tensor_spec.create %[[INDEX0]]
// CHECK-NEXT:   %[[INDEX1:.*]] = index.casts %[[CASTED1]] : si64 to index
// CHECK-NEXT:   %[[SPLIT_SPEC2:.*]] = mgp.tensor_spec.create %[[INDEX1]]
// CHECK-NEXT:   %[[SPLIT_SIZE1:.*]] = index.mul %[[INDEX0]], %[[THIRTYTWO]]

// CHECK-NEXT:   %[[SLICE:.*]] = mgp.buffer.slice %[[INPUT_BUF]], %[[ZERO]], %[[SPLIT_SIZE1]]

// CHECK-NEXT:   %[[SPLIT_SIZE2:.*]] = index.mul %[[INDEX1]], %[[THIRTYTWO]]
// CHECK-NEXT:   %[[SLICE2:.*]] = mgp.buffer.slice %[[INPUT_BUF]], %[[SPLIT_SIZE1]], %[[SPLIT_SIZE2]]

// CHECK-NEXT:   %[[SPLIT_TENSOR1:.*]] = mgp.tensor.create %[[CH]], %[[SLICE]], %[[SPLIT_SPEC1]]
// CHECK-NEXT:   %[[SPLIT_TENSOR2:.*]] = mgp.tensor.create %[[CH]], %[[SLICE2]], %[[SPLIT_SPEC2]]
// CHECK-NEXT:   mef.output %[[CH]], %[[SPLIT_TENSOR1]], %[[SPLIT_TENSOR2]]

// We pass in the output shape parameters of split to separate how we derived
// those parameters from the mogg e2e testing.
mo.graph @mogg_split<D0, D1, D2>(
  %input: !mo.tensor<[D0, 8], f32>,
  %dummy: !mo.tensor<[D1, D2], si64>
) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>) {

  %res:2 = mogg.split(%input) : (!mo.tensor<[D0, 8], f32>
  ) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>)

  mo.output %res#0, %res#1 : !mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>
}


// CHECK-LABEL: mgp.model @mogg_split_chain_prop : mof

// CHECK: } execute (%[[CH0:.*]]: !mef.chain,
// CHECK: %[[CH1:.*]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH0]]

// CHECK: %[[OUT_BUF1:.*]] = mgp.buffer.slice
// CHECK: %[[OUT_BUF2:.*]] = mgp.buffer.slice

// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH1]], {{.*}}%[[OUT_BUF1]]

// CHECK: mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH1]], {{.*}}%[[OUT_BUF2]]

mo.graph @mogg_split_chain_prop<D0, D1, D2>(
  %input: !mo.tensor<[D0, 8], f32>,
  %dummy: !mo.tensor<[D1, D2], si64>
) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>) {

  %0 = mo.relu(%input): !mo.tensor<[D0, 8], f32>

  %res:2 = mogg.split(%0) : (!mo.tensor<[D0, 8], f32>
  ) -> (!mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>)

  %1 = mo.relu(%res#0): !mo.tensor<[D1, 8], f32>
  %2 = mo.relu(%res#1): !mo.tensor<[D2, 8], f32>

  mo.output %1, %2 : !mo.tensor<[D1, 8], f32>, !mo.tensor<[D2, 8], f32>
}
