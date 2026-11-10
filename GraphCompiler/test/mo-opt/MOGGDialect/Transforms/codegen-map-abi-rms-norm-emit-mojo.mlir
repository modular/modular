// RUN: mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// With no kernel registered for it, rms_norm decomposes into its parts, which
// fuse back into one rowwise kernel shaped like the hand-written one: a cached
// bf16 row, a sum of squares in f32, the inverse rms once per row, and a map
// phase that scales by gamma and casts back, with the same launch parameters.
// CHECK-LABEL: def __static_broadcast_to_cast_add_{{[0-9a-f]+}}(
// CHECK:       var [[ROW:.*]] = algorithm.rowwise.Row[params, DType.float32, DType.bfloat16, 1, {{.*}}, is_cached = True](row_coords, layout.Idx[4096], row_ctx, [[LOAD:load_+[0-9]+]])
// CHECK-NEXT:  def {{contribute_+[0-9]+}}[w : Int](tile: SIMD[DType.bfloat16, w], {{.*}}) {var} -> SIMD[DType.float32, w]:
// CHECK-NEXT:    var [[A:.*]] = builtin_kernels.elementwise.Cast.elementwise[out_dtype = DType.float32](tile)
// CHECK-NEXT:    var [[SQ:.*]] = builtin_kernels.elementwise.Mul.elementwise[dtype = DType.float32]([[A]], [[A]])
// CHECK-NEXT:    return [[SQ]]
// CHECK:       var [[ACC:.*]] = [[ROW]].reduce[builtin_kernels.monoids.ReduceSum[dtype = DType.float32, W = params.simd_width]]
// CHECK:       var [[MS:.*]] = builtin_kernels.elementwise.Div.elementwise[dtype = DType.float32]([[ACC]].acc,
// CHECK-NEXT:  var [[V:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[MS]], SIMD[DType.float32, params.simd_width](9.99999997E-7))
// CHECK-NEXT:  var [[INV:.*]] = builtin_kernels.elementwise.Rsqrt.elementwise[dtype = DType.float32]([[V]])
// CHECK-NEXT:  def [[EW:elementwise_+[0-9]+]][w : Int](tile: SIMD[DType.bfloat16, w], {{.*}}) {var}:
// CHECK-NEXT:    comptime [[AL:.*]] = algorithm.rowwise_types.Context[params].element_alignment[DType.bfloat16, w]()
// CHECK-NEXT:    var [[X:.*]] = builtin_kernels.elementwise.Cast.elementwise[out_dtype = DType.float32](tile)
// CHECK-NEXT:    var [[NX:.*]] = builtin_kernels.elementwise.Mul.elementwise[dtype = DType.float32]([[X]], [[INV]].slice[w]())
// CHECK-NEXT:    var [[G:.*]] = managed_tensor_slice.simd_load_from_tensor_pointer[{{.*}}simd_width = w, element_alignment = [[AL]]]
// CHECK-NEXT:    var [[GF:.*]] = builtin_kernels.elementwise.Cast.elementwise[out_dtype = DType.float32]([[G]])
// CHECK-NEXT:    var [[GO:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[GF]], SIMD[DType.float32, w](0.0))
// CHECK-NEXT:    var [[Y:.*]] = builtin_kernels.elementwise.Mul.elementwise[dtype = DType.float32]([[NX]], [[GO]])
// CHECK-NEXT:    var [[OUT:.*]] = builtin_kernels.elementwise.Cast.elementwise[out_dtype = DType.bfloat16]([[Y]])
// CHECK-NEXT:    managed_tensor_slice.simd_store_into_tensor_pointer[{{.*}}simd_width = w, element_alignment = [[AL]]]({{.*}}[[OUT]]
// CHECK:       [[ROW]].elementwise([[EW]], [[LOAD]])
// CHECK:       builtin_primitives.primitives._rowwise[builtin_kernels.monoids.ReduceSum[dtype = DType.float32, W = 1], DType.bfloat16, DType.float32, axis = 1, num_phases = 2, computationally_expensive = True, dtype_size = 2]
// CHECK-NOT:   def __{{.*}}(arg
mo.graph @rmsNormBf16(%x: !mo.tensor<[8, 4096], bf16>, %w: !mo.tensor<[4096], bf16>) -> !mo.tensor<[8, 4096], bf16> {
  %y = mo.composite.reduce.rms_norm(%x, %w) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = true} : (!mo.tensor<[8, 4096], bf16>, !mo.tensor<[4096], bf16>) -> !mo.tensor<[8, 4096], bf16>
  mo.output %y : !mo.tensor<[8, 4096], bf16>
}
