// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --cse | FileCheck %s

// CHECK-LABEL: mgp.model @static_ranks_int_to_unknown : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<42x?xf32>)
// CHECK-DAG:   %[[CDIM:.*]] = index.constant 42
// CHECK-DAG:   %[[SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <42x?xf32>
// CHECK-DAG:   %[[DYNDIM:.*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <42x?xf32>
// CHECK-DAG:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[ARG2]] : <42x?xf32>
// CHECK-DAG:   %[[NEWSPEC:.*]] = mgp.tensor_spec.create %[[CDIM]], %[[DYNDIM]] : !mgp.tensor_spec<?x?xf32>
// CHECK-DAG:   %[[RES:.*]] = mgp.tensor.create %[[CH0]], %[[BUF]], %[[NEWSPEC]] : !mgp.tensor<?x?xf32>
// CHECK:   mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<?x?xf32>
// CHECK: }
mo.graph @static_ranks_int_to_unknown<D0, D1>(
  %arg0: !mo.tensor<[42, D0], f32>
) -> !mo.tensor<[?, D0], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[42, D0], f32> -> !mo.tensor<[?, D0], f32>
  mo.output %res : !mo.tensor<[?, D0], f32>
}

// CHECK-LABEL: mgp.model @static_ranks_unknown_to_int : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<?x?xf32>)
// CHECK:   %[[SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[ARG2]] : <?x?xf32>
// CHECK:   %[[DYNDIM1:.*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]] : <?x?xf32>
// CHECK:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[ARG2]] : <?x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.tensor_spec.create %[[DYNDIM1]] : !mgp.tensor_spec<42x?xf32>
// CHECK:   %[[RES:.*]] = mgp.tensor.create %[[CH0]], %[[BUF]], %[[NEWSPEC]]  : !mgp.tensor<42x?xf32>
// CHECK:   mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<42x?xf32>
// CHECK: }
mo.graph @static_ranks_unknown_to_int<D0, D1 -> ()>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, D0], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, D0], f32>
  mo.output %res : !mo.tensor<[42, D0], f32>
}

// CHECK-LABEL: mgp.model @static_ranks_unknown_to_static : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<?x?xf32>)
// CHECK:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[ARG2]] : <?x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.tensor_spec.create : !mgp.tensor_spec<42x10xf32>
// CHECK:   %[[RES:.*]] = mgp.tensor.create %[[CH0]], %[[BUF]], %[[NEWSPEC]] : !mgp.tensor<42x10xf32>
// CHECK:   mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<42x10xf32>
// CHECK: }
mo.graph @static_ranks_unknown_to_static<D0, D1>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, 10], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, 10], f32>
  mo.output %res : !mo.tensor<[42, 10], f32>
}

// CHECK-LABEL: mgp.model @multiple_rebind : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain,

// CHECK: %[[CH1:.*]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}(%[[CH0]]

// CHECK: %[[CH_M1:.*]] = grt.chain.create (%[[CH1]], %[[CH0]])

// CHECK: %[[CH_M2:.*]] = grt.chain.create (%[[CH_M1]], %[[CH0]])

// CHECK: %[[CH3:.*]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}({{.*}}%[[CH_M2]],

// CHECK: %[[CH_M3:.*]] = grt.chain.create (%[[CH3]], %[[CH0]])

// CHECK: %[[CH_M4:.*]] = grt.chain.create (%[[CH_M3]], %[[CH0]])

// CHECK: %[[CH5:.*]] = mogg.routine.async {{(dispatch\([a-z]+\) )?}}({{.*}}%[[CH_M4]]


mo.graph @multiple_rebind<D0, D1>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, 10], f32> {

  %0 = mo.relu(%arg0) : !mo.tensor<[D1, D0], f32>

  %1 = mo.rebind(%0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, 10], f32>

  %2 = mo.relu(%1) : !mo.tensor<[42, 10], f32>

  %3 = mo.rebind(%2) : !mo.tensor<[42, 10], f32> -> !mo.tensor<[D1, D0], f32>

  %4 = mo.relu(%3) : !mo.tensor<[D1, D0], f32>

  %5 = mo.rebind(%4) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, 10], f32>

  mo.output %5 : !mo.tensor<[42, 10], f32>
}
