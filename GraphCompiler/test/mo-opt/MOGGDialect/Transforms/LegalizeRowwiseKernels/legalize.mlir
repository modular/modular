// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels --map-rcse --map-outline-structured-kernel-bodies --map-to-mogg --mogg-legalize-rowwise-kernels | FileCheck %s

// A fused reduction, from MO to a rowwise kernel ready to print as Mojo.

// CHECK-LABEL: mo.graph @sumOfSquaresPlusBias
// CHECK:       mogg.call.rowwise {{.*}} attributes {accum = f32, cols = 128 : i64, dtype_size = 0 : i64, num_phases = 1 : i64, simd_monoid = @builtin_kernels.monoids.ReduceSum} {
// CHECK-NEXT:    mogg.rowwise.cache (%[[K:.*]] : !map.index<2>) -> !mogg.simd<bf16> {
// CHECK-NEXT:      %[[X:.*]] = mogg.tensor.load %cap0[%[[K]]]
// CHECK-NEXT:      mogg.yield %[[X]] : !mogg.simd<bf16>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[ACC:.*]] = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
// CHECK-NEXT:      %[[P:.*]] = mogg.rowwise.cache.load : !mogg.simd<bf16>
// CHECK-NEXT:      %[[A:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Cast(%[[P]])
// CHECK-NEXT:      %[[SQ:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Mul(%[[A]], %[[A]])
// CHECK-NEXT:      mogg.yield %[[SQ]] : !mogg.simd<f32>
// CHECK-NEXT:    } {fields = ["acc"], params = {dtype = f32}}
// CHECK-NEXT:    mogg.rowwise.emit {
// CHECK-NEXT:      %[[B:.*]] = mogg.tensor.load %cap1[%row]
// CHECK-NEXT:      %[[Y:.*]] = mogg.call.elementwise @builtin_kernels.elementwise.Add(%[[ACC]], %[[B]])
// CHECK-NEXT:      mogg.tensor.store %cap2[%row] = %[[Y]]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
mo.graph @sumOfSquaresPlusBias(%x: !mo.tensor<[8, 128], bf16>, %b: !mo.tensor<[8, 1], f32>) -> !mo.tensor<[8, 1], f32> {
  %a = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %sq = mo.mul(%a, %a) : !mo.tensor<[8, 128], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %y = mo.add(%r, %b) : !mo.tensor<[8, 1], f32>
  mo.output %y : !mo.tensor<[8, 1], f32>
}

// An exp in the fold makes the kernel computationally expensive.
// CHECK-LABEL: mo.graph @sumOfExp
// CHECK:       mogg.call.rowwise {{.*}} computationally_expensive,
// CHECK:         mogg.call.elementwise @builtin_kernels.elementwise.Exp
mo.graph @sumOfExp(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32> {
  %e = mo.exp(%x) : !mo.tensor<[8, 128], f32>
  %r = mo.reduce.add(%e) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  mo.output %r : !mo.tensor<[8, 1], f32>
}

// A bf16 rms-style body: the Row loads x itself, in bf16, and each phase casts
// it to f32; the cast isn't part of the load, which would widen the cached row.
// CHECK-LABEL: mo.graph @bf16Normalize
// CHECK:       mogg.call.rowwise
// CHECK-NEXT:    mogg.rowwise.cache (%{{.*}} : !map.index<2>) -> !mogg.simd<bf16> {
// CHECK-NEXT:      %[[X:.*]] = mogg.tensor.load
// CHECK-NEXT:      mogg.yield %[[X]] : !mogg.simd<bf16>
// CHECK-NEXT:    }
// CHECK:         mogg.rowwise.reduce
// CHECK-NEXT:      %[[T:.*]] = mogg.rowwise.cache.load : !mogg.simd<bf16>
// CHECK-NEXT:      mogg.call.elementwise @builtin_kernels.elementwise.Cast(%[[T]])
// CHECK:         mogg.rowwise.map
// CHECK-NEXT:      %[[T2:.*]] = mogg.rowwise.cache.load : !mogg.simd<bf16>
// CHECK-NEXT:      mogg.call.elementwise @builtin_kernels.elementwise.Cast(%[[T2]])
mo.graph @bf16Normalize(%x: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32> {
  %a = mo.cast(%x) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %sq = mo.mul(%a, %a) : !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%a, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// A fused bf16 product: the Row loads x * s, still bf16, once per element.
// CHECK-LABEL: mo.graph @productNormalize
// CHECK:       mogg.call.rowwise
// CHECK-NEXT:    mogg.rowwise.cache (%{{.*}} : !map.index<2>) -> !mogg.simd<bf16> {
// CHECK-NEXT:      mogg.tensor.load
// CHECK-NEXT:      mogg.tensor.load
// CHECK-NEXT:      mogg.call.elementwise @builtin_kernels.elementwise.Mul
// CHECK-NEXT:      mogg.yield
// CHECK-NOT:     mogg.tensor.load
// CHECK:         mogg.rowwise.map
mo.graph @productNormalize(%x: !mo.tensor<[8, 128], bf16>, %s: !mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32> {
  %p = mo.mul(%x, %s) : !mo.tensor<[8, 128], bf16>
  %a = mo.cast(%p) : (!mo.tensor<[8, 128], bf16>) -> !mo.tensor<[8, 128], f32>
  %sq = mo.mul(%a, %a) : !mo.tensor<[8, 128], f32>
  %sum = mo.reduce.add(%sq) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%sum) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%a, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}

// A producer that reads its element's index computes the same value at the
// same element, so it moves into the cache, at the cache's index.
// CHECK-LABEL: mo.graph @indexedProducer
// CHECK:         mogg.rowwise.cache (%[[I:[[:alnum:]_]+]] : !map.index<2>) -> !mogg.simd<f32> {
// CHECK:           mogg.call.elementwise @testing_kernels.testing_kernels.ElemwiseAddWithIndex(%{{.*}}, %[[I]])
// CHECK:         mogg.rowwise.reduce
// CHECK-NEXT:      mogg.rowwise.cache.load
mo.graph @indexedProducer(%x: !mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32> {
  %z = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 128], f32>
  %s = mo.reduce.add(%z) : (!mo.tensor<[8, 128], f32>) -> !mo.tensor<[8, 1], f32>
  %sb = mo.static.broadcast_to(%s) : !mo.tensor<[8, 1], f32> -> !mo.tensor<[8, 128], f32>
  %y = mo.div(%z, %sb) : !mo.tensor<[8, 128], f32>
  mo.output %y : !mo.tensor<[8, 128], f32>
}
