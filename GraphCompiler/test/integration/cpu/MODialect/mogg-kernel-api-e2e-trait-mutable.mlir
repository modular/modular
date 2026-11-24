// RUN: mo-opt %s -o %t.mlir --gc-pipeline="kernel-library-paths=%kernel_api_test_pkg" --allow-unregistered-dialect
// RUN: mt --execute --model-inputs=ones %t.mlir | FileCheck %s

// End-to-end proof of the trait MUTABLE-INPUT emit path. `trait_inplace_increment`
// declares its argument `Some[MutableInput & TileTensorable]` and mutates it in
// place (`x[i] += 1`). For a mutable_input role the lowering stamps a
// `MutableInput & TileTensorable` stub bound, so the emitted
// inline `_MoggDenseTensorArg` must conform to `MutableInput` (gated on its io),
// not merely `Input` -- otherwise the generated Mojo would fail to satisfy its
// own bound. With an all-ones input the kernel returns all 2.

!bt = !mo.buffer<[2, 2], f32>
!c = !mo.chain

// CHECK-LABEL: --- Running 'trait_inplace_increment_graph':
// CHECK{LITERAL}: 'trait_inplace_increment_graph' returned tensor([[2.0000, 2.0000],
// CHECK{LITERAL}: [2.0000, 2.0000]], dtype=f32, shape=[2,2])
mo.graph @trait_inplace_increment_graph(%buf: !bt) -> (!bt, !c) {
  %ch0 = mo.chain.create()
  %ch1 = mo.custom [symbol = "trait_inplace_increment", device = #M.device_ref<"cpu", 0>](%buf, %ch0) : (!bt, !c) -> (!c)
  mo.output %buf, %ch1 : !bt, !c
}
