// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-outline-structured-kernel-bodies --map-to-mogg --mogg-to-mgp --canonicalize -cse | FileCheck --enable-var-scope %s

// CHECK-LABEL: mgp.model @collect_chains
// CHECK:      execute{{.*}}[[ARG2:%.*]]: !mgp.tensor<?x?xf32>, [[ARG3:%.*]]: !mgp.tensor<?x?xf32>
// CHECK-DAG:  [[CH0:%.*]] = mgp.assert
// CHECK-DAG:  [[CH1:%.*]] = mgp.assert
// CHECK-DAG:  [[CH2:%.*]] = grt.chain.create ([[CH0]], [[CH1]])
// CHECK-DAG:  [[G:%.*]]:2 = mgp.guard [[CH2]], [[ARG2]]
// CHECK-DAG:  mef.output{{.*}} [[G]]#0, [[G]]#1
mo.graph @collect_chains<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>)
 -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>)
{
  kgen.param.declare CH0: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "D1 != D2" : !kgen.string}, identical(:scalar<si64> D1, D2))>
  kgen.param.declare CH1: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "D2 != D3" : !kgen.string}, identical(:scalar<si64> D2, D3))>

  %ch0 = mosh.param.to_value: !mo.chain = <CH0>
  %ch1 = mosh.param.to_value: !mo.chain = <CH1>

  %gc = mo.chain.create (%ch0, %ch1) : !mo.chain, !mo.chain

  %arg0p, %arg1p = mo.guard[%gc] (%arg0, %arg1) : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>

  mo.output %arg0p, %arg1p : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>
}
