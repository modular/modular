// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute   --input-shapes=3x2x1x4 --input-shapes=3x2x4x1 -model-inputs=ones %t.mlir | FileCheck --check-prefix=CHECK-WORKING %s

// RUN: mt --execute   --input-shapes=3x2x1x4 --input-shapes=3x2x2x1 -model-inputs=ones %t.mlir > %t.output1 || true
// RUN: mt --execute   --input-shapes=3x5x1x4 --input-shapes=3x2x4x1 -model-inputs=ones %t.mlir > %t.output2 || true
// RUN: mt --execute   --input-shapes=1x2x1x4 --input-shapes=3x2x4x1 -model-inputs=ones %t.mlir > %t.output3 || true

// RUN: cat %t.output1 | FileCheck --check-prefix=CHECK-INNER %s
// RUN: cat %t.output2 | FileCheck --check-prefix=CHECK-2ND %s
// RUN: cat %t.output3 | FileCheck --check-prefix=CHECK-1ST %s

// COM: check working example
//CHECK-WORKING{LITERAL}: --- Running 'bmm':
//CHECK-WORKING{LITERAL}: 'bmm' returned tensor([[[[4.0000]],
//CHECK-WORKING{LITERAL}: [[4.0000]],
//CHECK-WORKING{LITERAL}: [[4.0000]],
//CHECK-WORKING{LITERAL}: [[4.0000]],
//CHECK-WORKING{LITERAL}: [[4.0000]],
//CHECK-WORKING{LITERAL}: [[4.0000]]]], dtype=f32, shape=[3,2,1,1])

// COM: fail assertions on each possible pair of dimensions
// CHECK-INNER: 'bmm' returned <error:
// CHECK-INNER: inner dimension must match>
// CHECK-1ST: 'bmm' returned <error:
// CHECK-1ST: 1st batch dimensions must match>
// CHECK-2ND: 'bmm' returned <error:
// CHECK-2ND: 2nd batch dimensions must match>

mo.graph @bmm<D1, D2, D3, D4, D5, D6, D7, D8>(%arg0: !mo.tensor<[D1, D2, D3, D4], f32>, %arg1: !mo.tensor<[D5, D6, D7, D8], f32>) -> !mo.tensor<[D1, D2, D3, D8], f32> {

  kgen.param.declare CH0: !mo.chain = <apply(:(!kgen.scalar<bool>)            -> !mo.chain "mo.assert"{message = "1st batch dimensions must match" : !kgen.string},      identical(:scalar<si64> D1, D5))>
  kgen.param.declare CH1: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "2nd batch dimensions must match" : !kgen.string}, CH0, identical(:scalar<si64> D2, D6))>
  kgen.param.declare CH2: !mo.chain = <apply(:(!mo.chain, !kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "inner dimension must match" : !kgen.string},      CH1, identical(:scalar<si64> D4, D7))>

  %ch0 = mosh.param.to_value: !mo.chain = <CH0>
  %ch1 = mosh.param.to_value: !mo.chain = <CH2>
  %ch2 = mosh.param.to_value: !mo.chain = <CH1>

  %gc = mo.chain.create(%ch0, %ch1, %ch2) : !mo.chain, !mo.chain, !mo.chain

  %arg0p, %arg1p = mo.guard[%gc](%arg0, %arg1) : !mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D5, D6, D7, D8], f32>

  %0 = mo.batch_matmul(%arg0p, %arg1p) : (!mo.tensor<[D1, D2, D3, D4], f32>, !mo.tensor<[D5, D6, D7, D8], f32>) -> !mo.tensor<[D1, D2, D3, D8], f32>

  mo.output %0 : !mo.tensor<[D1, D2, D3, D8], f32>
}
