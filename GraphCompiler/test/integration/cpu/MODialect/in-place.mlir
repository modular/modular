// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=true fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=false fuse-jit-kernels=true" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=true fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// RUN: mo-opt --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg fuse-mutable-load-store=false fuse-jit-kernels=false" %s -o %t.mlir
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

!tt = !mo.tensor<[2, 2], f32>
!bt = !mo.buffer<[2, 2], f32>
!c = !mo.chain

// CHECK-LABEL: --- Running 'basic_add':
// CHECK{LITERAL}: 'basic_add' returned tensor([[2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000]], dtype=f32, shape=[2,2])
mo.graph @basic_add(%buf: !bt, %rhs: !tt) -> (!bt, !c)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %res = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%buf, %res, %ch1) : (!bt, !tt, !c) -> (!c)

  mo.output %buf, %ch2 : !bt, !c
}

// CHECK-LABEL: --- Running 'sequenced_adds':
// CHECK{LITERAL}: 'sequenced_adds' returned tensor([[3.0000, 3.0000],
// CHECK{LITERAL}: [3.0000, 3.0000]], dtype=f32, shape=[2,2])
mo.graph @sequenced_adds(%buf: !bt, %rhs: !tt) -> (!bt, !c)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !c) -> (!c)

  %vbuf2, %ch3 = mo.mutable.load(%buf, %ch2) : (!bt, !c) -> (!tt, !c)
  %res2 = mo.add(%vbuf2, %rhs) : !tt
  %ch4 = mo.mutable.store(%buf, %res2, %ch3) : (!bt, !tt, !c) -> (!c)

  mo.output %buf, %ch4 : !bt, !c
}

// CHECK-LABEL: --- Running 'add_relu':
// CHECK{LITERAL}: 'add_relu' returned tensor([[3.0000, 0.0000],
// CHECK{LITERAL}: [4.0000, 0.0000]], dtype=f32, shape=[2,2])
mo.graph @add_relu(%buf: !bt) -> (!bt, !c)
{
  %init_val = mo.constant <#M.dense_array<2.0, -2.0, 3.0, -4.0> : tensor<2x2xf32>> : !mo.tensor<[2, 2], f32>

  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %res1 = mo.add(%vbuf, %init_val) : !tt
  %ch2 = mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !c) -> (!c)

  %vbuf2, %ch3 = mo.mutable.load(%buf, %ch2) : (!bt, !c) -> (!tt, !c)
  %res2 = mo.relu(%vbuf2) : !tt
  %ch4 = mo.mutable.store(%buf, %res2, %ch3) : (!bt, !tt, !c) -> (!c)

  mo.output %buf, %ch4 : !bt, !c
}

// CHECK-LABEL: --- Running 'add_neg':
// CHECK{LITERAL}: 'add_neg' returned tensor([[-2.0000, -2.0000],
// CHECK{LITERAL}: [-2.0000, -2.0000]], dtype=f32, shape=[2,2])
mo.graph @add_neg(%buf: !bt, %rhs: !tt) -> (!bt, !c)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !c) -> (!c)

  %res2 = mo.negative(%res1) : !tt
  %ch4 = mo.mutable.store(%buf, %res2, %ch2) : (!bt, !tt, !c) -> (!c)

  mo.output %buf, %ch4 : !bt, !c
}

// CHECK-LABEL: --- Running 'load_value_multiple_uses':
// CHECK{LITERAL}: 'load_value_multiple_uses' returned tensor([[3.0000, 3.0000],
// CHECK{LITERAL}: [3.0000, 3.0000]], dtype=f32, shape=[2,2])
// CHECK{LITERAL}: , tensor([[-1.0000, -1.0000],
// CHECK{LITERAL}: [-1.0000, -1.0000]], dtype=f32, shape=[2,2])
mo.graph @load_value_multiple_uses(%buf: !bt, %rhs: !tt) -> (!bt, !tt, !c)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !c) -> (!c)

  %vbuf2, %ch3 = mo.mutable.load(%buf, %ch2) : (!bt, !c) -> (!tt, !c)
  %res2 = mo.add(%vbuf2, %rhs) : !tt
  %ch4 = mo.mutable.store(%buf, %res2, %ch3) : (!bt, !tt, !c) -> (!c)

  %r1 = mo.negative(%vbuf) : !tt

  mo.output %buf, %r1, %ch4 : !bt, !tt, !c
}
