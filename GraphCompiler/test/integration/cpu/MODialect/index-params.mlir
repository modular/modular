// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=1x10x2 -model-inputs=zeros --result-output-style=full %t.mlir | FileCheck %s

mo.graph @range_with_runtime_index_si32<START, LIMIT, STEP -> N>(
  %arg0: !mo.tensor<[START, LIMIT, STEP], f32>
) -> !mo.tensor<[N], si32> {
  %start = mosh.param.to_value : !kgen.scalar<si64> = <START>
  %limit = mosh.param.to_value : !kgen.scalar<si64> = <LIMIT>
  %step = mosh.param.to_value : !kgen.scalar<si64> = <STEP>
  %startT = mo.index.to_tensor(%start) -> !mo.tensor<[], si32>
  %limitT = mo.index.to_tensor(%limit) -> !mo.tensor<[], si32>
  %stepT = mo.index.to_tensor(%step) -> !mo.tensor<[], si32>
  %res = mo.range<() -> D0>(%startT, %limitT, %stepT) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[D0], si32>
  %resreb = mo.rebind<() -> M>(%res) : !mo.tensor<[D0], si32> -> !mo.tensor<[M], si32>
  mo.output<M> %resreb : !mo.tensor<[M], si32>
}
// CHECK{LITERAL}: 'range_with_runtime_index_si32' returned tensor<5xsi32> [1, 3, 5, 7, 9]

mo.graph @range_with_runtime_index_si64<START, LIMIT, STEP -> N>(
  %arg0: !mo.tensor<[START, LIMIT, STEP], f32>
) -> !mo.tensor<[N], si64> {
  %start = mosh.param.to_value : !kgen.scalar<si64> = <START>
  %limit = mosh.param.to_value : !kgen.scalar<si64> = <LIMIT>
  %step = mosh.param.to_value : !kgen.scalar<si64> = <STEP>
  %startT = mo.index.to_tensor(%start) -> !mo.tensor<[], si64>
  %limitT = mo.index.to_tensor(%limit) -> !mo.tensor<[], si64>
  %stepT = mo.index.to_tensor(%step) -> !mo.tensor<[], si64>
  %res = mo.range<() -> D0>(%startT, %limitT, %stepT) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D0], si64>
  %resreb = mo.rebind<() -> M>(%res) : !mo.tensor<[D0], si64> -> !mo.tensor<[M], si64>
  mo.output<M> %resreb : !mo.tensor<[M], si64>
}
// CHECK{LITERAL}: 'range_with_runtime_index_si64' returned tensor<5xsi64> [1, 3, 5, 7, 9]
