// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-fuse-mutable-loads --map-allocate-accumulating-buffers --map-to-mogg | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-fuse-mutable-loads --map-allocate-accumulating-buffers --map-to-mogg="enable-compute-lambdas=false" | FileCheck %s --check-prefix=NOCOMPUTE

// `trait_fused_transform_bias` declares its output as a trait-lane
// `_LegacyComputeOutput` slot (contract access `transform`). It follows the
// same rules as a legacy `_FusedComputeOutputTensor`: the epilogue becomes a
// compute lambda (a yield-terminated body) when it can, and is downgraded to a
// store lambda otherwise. Emit wraps a yield body as `_MoggTransformFusionTensor`
// and a store body as `_MoggOutputStoreFusionTensor`.

// A same-dtype elementwise epilogue is a compute lambda.
// CHECK-LABEL: mo.graph @slot_add
// CHECK:       mogg.tensor.create.epilogue {{.*}}computeOutputArg = true contract = <{role = output, fused = true, access = transform, capabilities = none}>
// CHECK-NOT:     mogg.tensor.store
// CHECK:         mogg.yield
// With compute lambdas off, the same epilogue is downgraded to a store.
// NOCOMPUTE-LABEL: mo.graph @slot_add
// NOCOMPUTE:       mogg.tensor.create.epilogue {{.*}}access = transform
// NOCOMPUTE:         mogg.tensor.store
// NOCOMPUTE-NOT:     mogg.yield
// NOCOMPUTE:       mo.graph @slot_no_epilogue
mo.graph @slot_add(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%z, %y) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}

// With nothing fused, the slot still gets the compute form, with an identity
// body.
// CHECK-LABEL: mo.graph @slot_no_epilogue
// CHECK:       mogg.tensor.create.epilogue %{{[^ ]+}} (%[[V:[^ ]+]] : !mogg.simd<f32>, %{{[^ ]+}} : !map.index<1>) {{.*}}access = transform
// CHECK-NEXT:    mogg.yield %[[V]] : !mogg.simd<f32>
mo.graph @slot_no_epilogue(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %z : !mo.tensor<[4], f32>
}

// A dtype-changing epilogue is downgraded: `compute` returns the dtype it was
// handed, so the cast has to happen in a store.
// CHECK-LABEL: mo.graph @slot_cast
// CHECK:       mogg.tensor.create.epilogue %{{[^ ]+}} (%{{[^ ]+}} : !mogg.simd<f32>, %[[I:[^ ]+]] : !map.index<1>) capturing {%[[C:[^ ]+]] = {{.*}}access = transform{{.*}}elementType = f64
// CHECK:         mogg.tensor.store %[[C]][%[[I]]] = %{{[^ ]+}} : !mogg.tensor, !map.index<1>, !mogg.simd<f64>
// CHECK-NOT:     mogg.yield
mo.graph @slot_cast(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f64> {
  %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.cast(%z) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f64>
  mo.output %out : !mo.tensor<[4], f64>
}

// A store-index transform (a fused reshape) is downgraded: a compute lambda's
// value is stored at the kernel's own index, which would drop the reshape.
// CHECK-LABEL: mo.graph @slot_reshape
// CHECK:       mogg.tensor.create.epilogue {{.*}}access = transform
// CHECK:         %[[SIDX:.*]] = mogg.index.reshape
// CHECK:         mogg.tensor.store %{{[^ ]+}}[%[[SIDX]]]
// CHECK-NOT:     mogg.yield
mo.graph @slot_reshape(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %r = mo.static.reshape(%z) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 2], f32>
  %out = mo.add(%r, %y) : !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}

// Classification resolves the kernel argument through the operand mapping, not
// the raw result position. Result 1 of `trait_tuple_then_compute_output` is
// tuple element 1, whose argument is the store-contract tuple; at raw
// position 1 sits the `_LegacyComputeOutput` slot `z`. The tuple element's
// epilogue stays a store lambda, and `z`'s still becomes a compute lambda.
// CHECK-LABEL: mo.graph @tuple_before_slot
// CHECK:       mogg.tensor.create.epilogue {{.*}}contract = <{role = output, fused = true, access = store, capabilities = none}>
// CHECK-NOT:     mogg.yield
// CHECK:         elementwise.Add
// CHECK-NEXT:    mogg.tensor.store
// CHECK:       mogg.tensor.create.epilogue {{.*}}computeOutputArg = true contract = <{role = output, fused = true, access = transform, capabilities = none}>
// CHECK-NOT:     mogg.tensor.store
// CHECK:         mogg.yield
mo.graph @tuple_before_slot(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %a, %b, %z = mo.custom [symbol = "trait_tuple_then_compute_output", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>)
  %b2 = mo.add(%b, %y) : !mo.tensor<[4], f32>
  %z2 = mo.negative(%z) : !mo.tensor<[4], f32>
  mo.output %a, %b2, %z2 : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

// An input fed by a fused `mo.mutable.load` reaches the call as a buffer. It
// still maps onto the read-only `x`, so the slot's epilogue stays a compute
// lambda.
// CHECK-LABEL: mo.graph @slot_mutable_load_input
// CHECK-NOT:   mo.mutable.load
// CHECK:       mogg.tensor.create.epilogue {{.*}}computeOutputArg = true contract = <{role = output, fused = true, access = transform, capabilities = none}>
// CHECK-NOT:     mogg.tensor.store
// CHECK:         mogg.yield
mo.graph @slot_mutable_load_input(%buf: !mo.buffer<[4], f32>, %y: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %ch0 = mo.chain.create()
  %x, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[4], f32>, !mo.chain) -> (!mo.tensor<[4], f32>, !mo.chain)
  %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.add(%z, %y) : !mo.tensor<[4], f32>
  mo.output %out : !mo.tensor<[4], f32>
}

// An epilogue writing two outputs is downgraded: a compute lambda returns one
// value for one destination.
// CHECK-LABEL: mo.graph @slot_two_outputs
// CHECK:       mogg.tensor.create.epilogue {{.*}}access = transform
// CHECK:         mogg.tensor.store
// CHECK:         mogg.tensor.store
// CHECK-NOT:     mogg.yield
// CHECK:       mogg.primitive.decl
mo.graph @slot_two_outputs(%x: !mo.tensor<[4], f32>, %y: !mo.tensor<[4], f32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) {
  %z = mo.custom [symbol = "trait_fused_transform_bias", device = #M.device_ref<"cpu", 0>] (%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %a = mo.add(%z, %y) : !mo.tensor<[4], f32>
  %b = mo.mul(%z, %y) : !mo.tensor<[4], f32>
  mo.output %a, %b : !mo.tensor<[4], f32>, !mo.tensor<[4], f32>
}

