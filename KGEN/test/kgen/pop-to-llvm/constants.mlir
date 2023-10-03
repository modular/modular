// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// COM: Checking the LLVMIR is easier since the constants are collapsed.

// CHECK-LABEL: @array_constant
kgen.func @array_constant() -> !pop.array<2, i32> {
  // CHECK-NEXT: [2 x i32] [i32 1, i32 2]
  %0 = kgen.param.constant: array<2, i32> = <[1, 2]>
  kgen.return %0 : !pop.array<2, i32>
}

// CHECK-LABEL: @struct_constant
kgen.func @struct_constant() -> !pop.struct<array<1, i32>, struct<i32, i32>> {
  // CHECK-NEXT: { [1 x i32], { i32, i32 } }
  // CHECK-SAME: { [1 x i32] [i32 1], { i32, i32 } { i32 2, i32 3 } }
  %0 = kgen.param.constant: struct<array<1, i32>, struct<i32, i32>> =
    <{ [1], { 2, 3 } }>
  kgen.return %0 : !pop.struct<array<1, i32>, struct<i32, i32>>
}

// CHECK-LABEL: @simd_constant
kgen.func @simd_constant() -> (!pop.simd<2, bool>, !pop.simd<2, si8>, !pop.scalar<bf16>) {
  // CHECK-NEXT: { <2 x i1>, <2 x i8>, bfloat }
  // CHECK-SAME: { <2 x i1> <i1 true, i1 false>, <2 x i8> <i8 -3, i8 3>, bfloat 0xR3FA0 }
  %0 = kgen.param.constant: simd<2, bool> = <<true, false>>
  %1 = kgen.param.constant: simd<2, si8> = <<-3, 3>>
  %2 = kgen.param.constant: scalar<bf16> = <<"1.25">>
  kgen.return %0, %1, %2 : !pop.simd<2, bool>, !pop.simd<2, si8>, !pop.scalar<bf16>
}

// CHECK-LABEL: @scalar_index_addr_constants
kgen.func @scalar_index_addr_constants() -> (!pop.scalar<index>, !pop.scalar<address>) {
  // CHECK-NEXT: { i64, ptr } { i64 1, ptr inttoptr (i64 2 to ptr) }
  %0 = kgen.param.constant: scalar<index> = <<1>>
  %1 = kgen.param.constant: scalar<address> = <<2>>
  kgen.return %0, %1 : !pop.scalar<index>, !pop.scalar<address>
}

// CHECK-LABEL: @simd_index_addr_constants
kgen.func @simd_index_addr_constants() -> (!pop.simd<2, index>, !pop.simd<2, address>) {
  // CHECK-NEXT: { <2 x i64>, <2 x ptr> }
  // CHECK-SAME: { <2 x i64> <i64 1, i64 11>, <2 x ptr> <ptr inttoptr (i64 2 to ptr), ptr inttoptr (i64 22 to ptr)> }
  %0 = kgen.param.constant: simd<2, index> = <<1, 11>>
  %1 = kgen.param.constant: simd<2, address> = <<2, 22>>
  kgen.return %0, %1 : !pop.simd<2, index>, !pop.simd<2, address>
}

// CHECK-LABEL: @variant_constant_0
kgen.func @variant_constant_0() -> !pop.variant<i32> {
  // CHECK-NEXT: { [1 x i64], i1 } { [1 x i64] [i64 1], i1 false }
  %0 = kgen.param.constant: variant<i32> = <#pop.variant<:i32 1, 0>>
  kgen.return %0 : !pop.variant<i32>
}

// CHECK-LABEL: @variant_constant_1
kgen.func @variant_constant_1() -> !pop.variant<struct<i32, i64, i32>, struct<f64, f32>> {
  // CHECK-NEXT: { [2 x i64], i1 } { [2 x i64] [i64 8589934593, i64 12884901888], i1 false }
  %0 = kgen.param.constant: variant<struct<i32, i64, i32>, struct<f64, f32>> = <#pop.variant<:!pop.struct<i32, i64, i32> { 1, 2, 3 }, 0>>
  kgen.return %0 : !pop.variant<struct<i32, i64, i32>, struct<f64, f32>>
}

// CHECK-LABEL: @variant_constant_2
kgen.func @variant_constant_2() -> !pop.variant<i1, i2, i3, i4, i5, i6> {
  // CHECK-NEXT: { [1 x i64], i3 } { [1 x i64] [i64 1], i3 3 }
  %0 = kgen.param.constant: variant<i1, i2, i3, i4, i5, i6> = <#pop.variant<:i4 1, 3>>
  kgen.return %0 : !pop.variant<i1, i2, i3, i4, i5, i6>
}

// CHECK-LABEL: @variadic_constant_0
kgen.func @variadic_constant_0() -> !kgen.variadic<i1> {
  // CHECK: %[[ALLOCA:.*]] = alloca i1, i64 3, align 1
  // CHECK: insertvalue { ptr, i64 } undef, ptr %[[ALLOCA]], 0
  // CHECK: insertvalue { ptr, i64 } %{{[0-9]+}}, i64 3, 1
  %0 = kgen.param.constant: !kgen.variadic<i1> = <#kgen.variadic<0, 1, 0>>
  kgen.return %0 : !kgen.variadic<i1>
}

// CHECK-LABEL: @variadic_constant_1
kgen.func @variadic_constant_1() -> !kgen.variadic<i32> {
  // CHECK: %[[ALLOCA:.*]] = alloca i32, i64 0, align 4
  // CHECK: insertvalue { ptr, i64 } undef, ptr %[[ALLOCA]], 0
  // CHECK: insertvalue { ptr, i64 } %{{[0-9]+}}, i64 0, 1
  %0 = kgen.param.constant: !kgen.variadic<i32> = <#kgen.variadic<>>
  kgen.return %0 : !kgen.variadic<i32>
}

// CHECK-LABEL: @pack_constant_0
kgen.func @pack_constant_0() -> !pop.pack<[i32, i8]> {
  // CHECK-NEXT: { i32, i8 } { i32 1, i8 2 }
  %0 = kgen.param.constant: !pop.pack<[i32, i8]> = <<1, 2>>
  kgen.return %0 : !pop.pack<[i32, i8]>
}

// CHECK-LABEL: @pack_constant_1
kgen.func @pack_constant_1() -> !pop.pack<[]> {
  // CHECK-NEXT: {} undef
  %0 = kgen.param.constant: !pop.pack<[]> = <<>>
  kgen.return %0 : !pop.pack<[]>
}

// CHECK-LABEL: @pointer_constant
kgen.func @pointer_constant() -> !kgen.pointer<?> {
  // CHECK-NEXT: ptr null
  %null = kgen.param.constant: pointer<?> = <#interp.pointer<0>>
  kgen.return %null : !kgen.pointer<?>
}

}
