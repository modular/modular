// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-epilogue | FileCheck %s

// A `mo.mutable.store` consumer of a fused-output kernel looks like any other
// epilogue candidate -- single-use producer result, one-result consumer -- but
// `lowerMutableStoreToForeach` gives it a `!mo.chain` result and a body with no
// output block argument, the buffer input being the store target. It fuses the
// way MOGG's `FuseEpilogues` does: the buffer arrives as an epilogue capture
// like any other consumer input, and the epilogue's trailing store is rebuilt
// on it, so the kernel writes the buffer directly and its own tensor output
// goes unread.

// CHECK-LABEL: mo.graph @mutable_store_consumer_fused
// CHECK:       map.iter.opaque
// CHECK:       inputs (%{{.*}}: !mo.tensor<[4], f32>, %[[BUF:[a-zA-Z0-9_]+]]: !mo.buffer<[4], f32>)
// CHECK:       map.iter.epilogue
// CHECK-SAME:  %[[BCAP:[a-zA-Z0-9_]+]] = %[[BUF]] : !mo.buffer<[4], f32>}
// CHECK:       map.call.elementwise @{{.*}}MutableStore
// CHECK:       map.store %[[BCAP]]
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// The kernel now writes memory no result names, so it needs the write effect
// to survive DCE -- without it the store is dropped and the buffer never
// written, which is what used to happen.
// CHECK:       hasWriteEffect = true
// CHECK-NOT:   map.iter.elementwise
mo.graph @mutable_store_consumer_fused(
  %buf: !mo.buffer<[4], f32>,
  %a: !mo.tensor<[4], f32>
) {
  %chain = mo.chain.create()
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.mutable.store(%buf, %c, %chain) : (!mo.buffer<[4], f32>, !mo.tensor<[4], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// Declined when the store's chain is consumed: `map.iter.opaque` holds no
// chain of its own, so the ordering the consumer asked for has nowhere to go.

// CHECK-LABEL: mo.graph @chain_consumed_not_fused
// The kernel keeps its own output and its plain store epilogue.
// CHECK:       map.iter.epilogue
// CHECK-SAME:  %[[CAP:[a-zA-Z0-9_]+]] = %[[OUT:[a-zA-Z0-9_]+]] : !mo.tensor<[4], f32>}
// CHECK:       map.store %[[CAP]][%{{.*}}] = %{{.*}} : !mo.tensor<[4], f32>
// The store stays a separate kernel, writing the buffer itself.
// CHECK:       map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}MutableStore
mo.graph @chain_consumed_not_fused(
  %buf: !mo.buffer<[4], f32>,
  %a: !mo.tensor<[4], f32>
) -> !mo.chain {
  %chain = mo.chain.create()
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %out = mo.mutable.store(%buf, %c, %chain) : (!mo.buffer<[4], f32>, !mo.tensor<[4], f32>, !mo.chain) -> !mo.chain
  mo.output %out : !mo.chain
}

// Declined when the store is ordered after another one: the store's input
// chain would be dropped with the ordering it carries.
// CHECK-LABEL: mo.graph @store_after_store_not_fused
// CHECK:       map.iter.opaque
// CHECK:       map.iter.elementwise
// CHECK:       map.call.elementwise @{{.*}}MutableStore
mo.graph @store_after_store_not_fused(
  %buf: !mo.buffer<[4], f32>,
  %a: !mo.tensor<[4], f32>,
  %b: !mo.tensor<[4], f32>
) {
  %c0 = mo.chain.create()
  %c1 = mo.mutable.store(%buf, %b, %c0) : (!mo.buffer<[4], f32>, !mo.tensor<[4], f32>, !mo.chain) -> !mo.chain
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %c2 = mo.mutable.store(%buf, %c, %c1) : (!mo.buffer<[4], f32>, !mo.tensor<[4], f32>, !mo.chain) -> !mo.chain
  mo.output
}
