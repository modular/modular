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

// CHECK-LABEL: --- Running 'custom_separate_buffers':
// CHECK{LITERAL}: 'custom_separate_buffers' returned tensor([[0.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000]], dtype=f32, shape=[2,2])
// CHECK{LITERAL}: , tensor([[0.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000]], dtype=f32, shape=[2,2])
mo.graph @custom_separate_buffers(%b: !bt, %c: !bt) -> (!bt, !bt, !c) {
  %ch0 = mo.chain.create()

  %ch1 = mo.custom [symbol = "mutable_test_op", device = #M.device_ref<"cpu", 0>](%b, %ch0) : (!bt, !c) -> (!c)
  %ch2 = mo.custom [symbol = "mutable_test_op", device = #M.device_ref<"cpu", 0>](%c, %ch1) : (!bt, !c) -> (!c)

  mo.output %b, %c, %ch2 : !bt, !bt, !c
}

// CHECK-LABEL: --- Running 'custom_with_standard':
// CHECK{LITERAL}: 'custom_with_standard' returned tensor([[0.0000, -2.0000],
// CHECK{LITERAL}: [-2.0000, -2.0000]], dtype=f32, shape=[2,2])
mo.graph @custom_with_standard(%buf: !bt, %rhs: !tt) -> (!bt, !c)
{
  %ch0 = mo.chain.create(%buf) : !bt

  %vbuf, %ch1 = mo.mutable.load(%buf, %ch0) : (!bt, !c) -> (!tt, !c)
  %res1 = mo.add(%vbuf, %rhs) : !tt
  %ch2 = mo.mutable.store(%buf, %res1, %ch1) : (!bt, !tt, !c) -> (!c)

  %res2 = mo.negative(%res1) : !tt
  %ch4 = mo.mutable.store(%buf, %res2, %ch2) : (!bt, !tt, !c) -> (!c)

  %ch5 = mo.custom [symbol = "mutable_test_op", device = #M.device_ref<"cpu", 0>](%buf, %ch4) : (!bt, !c) -> (!c)

  mo.output %buf, %ch5 : !bt, !c
}

// CHECK-LABEL: --- Running 'custom_new_api':
// CHECK{LITERAL}: 'custom_new_api' returned tensor([[4.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000]], dtype=f32, shape=[2,2])
mo.graph @custom_new_api(%buf: !bt) -> (!bt, !c)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom [symbol = "inplace_increment_elem", device = #M.device_ref<"cpu", 0>](%buf, %ch0) : (!bt, !c) -> (!c)
  %ch2 = mo.custom [symbol = "inplace_increment_elem", device = #M.device_ref<"cpu", 0>](%buf, %ch1) : (!bt, !c) -> (!c)
  %ch3 = mo.custom [symbol = "inplace_increment_elem", device = #M.device_ref<"cpu", 0>](%buf, %ch2) : (!bt, !c) -> (!c)

  mo.output %buf, %ch3 : !bt, !c
}

// CHECK-LABEL: --- Running 'custom_new_api_raises':
// CHECK{LITERAL}: 'custom_new_api_raises' returned tensor([[4.0000, 1.0000],
// CHECK{LITERAL}: [1.0000, 1.0000]], dtype=f32, shape=[2,2])
mo.graph @custom_new_api_raises(%buf: !bt) -> (!bt, !c)
{
  %ch0 = mo.chain.create()

  %ch1 = mo.custom [symbol = "inplace_increment_elem_raises", device = #M.device_ref<"cpu", 0>](%buf, %ch0) : (!bt, !c) -> (!c)
  %ch2 = mo.custom [symbol = "inplace_increment_elem_raises", device = #M.device_ref<"cpu", 0>](%buf, %ch1) : (!bt, !c) -> (!c)
  %ch3 = mo.custom [symbol = "inplace_increment_elem_raises", device = #M.device_ref<"cpu", 0>](%buf, %ch2) : (!bt, !c) -> (!c)

  mo.output %buf, %ch3 : !bt, !c
}
