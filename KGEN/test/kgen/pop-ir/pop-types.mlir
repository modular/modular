// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @pointer
kgen.generator @pointer<ty: type, address_space>(
  // CHECK-SAME: !kgen.pointer<scalar<f32>>
  %arg0: !kgen.pointer<scalar<f32>>,
  // CHECK-SAME: !kgen.pointer<scalar<f32>, 5>
  %arg1: !kgen.pointer<scalar<f32>, 5>,
  // CHECK-SAME: !kgen.pointer<ty>
  %arg2: !kgen.pointer<ty>,
  // CHECK-SAME: !kgen.pointer<ty, 7>
  %arg3: !kgen.pointer<ty, 7>,
  // CHECK-SAME: !kgen.pointer<scalar<f32>, address_space>
  %arg4: !kgen.pointer<scalar<f32>, address_space>,
  // CHECK-SAME: !kgen.pointer<ty, address_space>
  %arg5: !kgen.pointer<ty, address_space>
) {
  kgen.return
}

// CHECK-LABEL: @array
kgen.generator @array<size, ty: type>(
  // CHECK-SAME: !pop.array<4, scalar<f32>>
  %arg0: !pop.array<4, scalar<f32>>,
  // CHECK-SAME: !pop.array<size, ty>
  %arg1: !pop.array<size, ty>
) {
  kgen.return
}

// CHECK-LABEL: @struct
kgen.generator @struct<size, dtype: dtype, ty: type>(
  // CHECK-SAME: !pop.struct<scalar<f32>, simd<4, ui64>>
  %arg0: !pop.struct<scalar<f32>, simd<4, ui64>>,
  // CHECK-SAME: !pop.struct<pointer<simd<4, si8>>, array<24, scalar<si64>>, struct<scalar<f32>, scalar<f64>>>
  %arg1: !pop.struct<
    !kgen.pointer<simd<4, si8>>,
    !pop.array<24, scalar<si64>>,
    !pop.struct<
      !pop.scalar<f32>,
      !pop.scalar<f64>
    >
  >,
  // CHECK: !pop.struct<ty, array<size, scalar<dtype>>>
  %arg2: !pop.struct<ty, array<size, scalar<dtype>>>
) {
  kgen.return
}

// CHECK-LABEL: @pack
kgen.generator @pack<Ts: variadic<!kgen.mlirtype>, T0: type, T1: type>(
  // CHECK-SAME: !pop.pack<Ts>
  %arg0: !pop.pack<Ts>,
  // CHECK-SAME: !pop.pack<[T0, T1]>
  %arg1: !pop.pack<[T0, T1]>,
  // CHECK-SAME: !pop.pack<[]>
  %arg2: !pop.pack<[]>,
  // CHECK-SAME: !pop.pack<[i32, i64]>
  %arg3: !pop.pack<[i32, i64]>
) {
  kgen.return
}

// CHECK-LABEL: @variadic
kgen.generator @variadic<ty: type>(
  // CHECK-SAME: !kgen.variadic<scalar<f32>>
  %arg0: !kgen.variadic<!pop.scalar<f32>>,
  // CHECK-SAME: !kgen.variadic<ty>
  %arg1: !kgen.variadic<ty>
) {
  kgen.return
}
