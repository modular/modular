// RUN: mo-opt --gc-pipeline %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-mutable-load-store=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: --- Running 'basic_if_tb':
// CHECK{LITERAL}: 'basic_if_tb' returned tensor([[2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000]], dtype=f32, shape=[2,2])

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!condt = !mo.tensor<[], bool>
!c = !mo.chain
mo.graph @basic_if_tb(%buf: !bt, %true: !condt) -> (!bt, !c)
{
  %ch0 = mo.chain.create()
  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)

  %och = mo.if(%true : !condt) -> !c {
    %1 = mo.add(%vbuf, %vbuf) : !tt
    %ch2_0 = mo.mutable.store(%buf, %1, %ch1) : (!bt, !tt, !c) -> !c
    mo.yield %ch2_0 : !c
  } else {
    %2 = mo.negative(%vbuf) : !tt
    %ch2_0 = mo.mutable.store(%buf, %2, %ch1) : (!bt, !tt, !c) -> !c
    mo.yield %ch2_0 : !c
  }

  mo.output %buf, %och : !bt, !c
}

// CHECK-LABEL: --- Running 'basic_if_fb':
// CHECK{LITERAL}: 'basic_if_fb' returned tensor([[-1.0000, -1.0000],
// CHECK{LITERAL}: [-1.0000, -1.0000]], dtype=f32, shape=[2,2])
mo.graph @basic_if_fb(%buf: !bt, %true: !condt) -> (!bt, !c)
{
  %false = mo.xor(%true, %true) : !condt

  %ch0 = mo.chain.create()
  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)

  %och = mo.if(%false : !condt) -> !c {
    %1 = mo.add(%vbuf, %vbuf) : !tt
    %ch2_0 = mo.mutable.store(%buf, %1, %ch1) : (!bt, !tt, !c) -> !c
    mo.yield %ch2_0 : !c
  } else {
    %2 = mo.negative(%vbuf) : !tt
    %ch2_0 = mo.mutable.store(%buf, %2, %ch1) : (!bt, !tt, !c) -> !c
    mo.yield %ch2_0 : !c
  }

  mo.output %buf, %och : !bt, !c
}

// CHECK-LABEL: --- Running 'basic_while':
// CHECK{LITERAL}: 'basic_while' returned tensor([[8.0000, 8.0000],
// CHECK{LITERAL}: [8.0000, 8.0000]], dtype=f32, shape=[2,2])
mo.graph @basic_while(%buf: !bt) -> (!bt, !c) {
   %ch0 = mo.chain.create()

   %x = mo.constant <#M.dense_array<1.0, 1.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
   %y = mo.constant <#M.dense_array<4.0> : tensor<1xf32>> : !mo.tensor<[], f32>

   // Need this slightly weird structure of passing in the chain like this due
   // to the structure of a mo.while op -- the signature of the shadowed
   // variables must match that of the ultimate output of the while.
   //
   // Really any chain could be passed in for our purposes here
   // (embedding in-place ops in a while) since mo.while isn't IsolatedFromAbove.
   %while_res:2 = mo.while (%x as %xi: !mo.tensor<[2], f32>, %ch0 as %inChain: !mo.chain) {
     %sum0 = mo.reduce.add(%xi) : (!mo.tensor<[2], f32>) -> !mo.tensor<[1], f32>
     %len = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
     %mean0 = mo.div(%sum0, %len) : !mo.tensor<[1], f32>
     %mean1 = mo.static.reshape(%mean0) : !mo.tensor<[1], f32> -> !mo.tensor<[], f32>
     %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.while.condition(%cond) %xi, %inChain : !mo.tensor<[2], f32>, !mo.chain
   } do {
     %tensor, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
     %res = mo.add(%tensor, %tensor) : !tt
     %ch2 = mo.mutable.store(%buf, %res, %ch1) : (!bt, !tt, !c) -> (!c)

     %new_x = mo.add (%xi, %xi) : !mo.tensor<[2], f32>
     mo.yield %new_x, %ch2 : !mo.tensor<[2], f32>, !mo.chain
   }

   %tensor, %ch1 = mo.mutable.load(%buf, %while_res#1) : (!bt, !c) -> (!tt, !c)
   %res = mo.add(%tensor, %tensor) : !tt
   %ch2 = mo.mutable.store(%buf, %res, %ch1) : (!bt, !tt, !c) -> (!c)

   mo.output %buf, %ch2 : !bt, !c
}
