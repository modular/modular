// RUN: kgen-opt -lower-kgen-to-llvm -split-input-file %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="skylake-avx512", features="+fma", data_layout="", simd_bit_width=128, tune_cpu="skylake-avx512">} {

// CHECK-LABEL: llvm.func internal @trivial
// CHECK-SAME: (%[[ARG0:.*]]: i32
// CHECK-SAME: ["target-cpu", "skylake-avx512"]
// CHECK-SAME: ["target-features", "+fma"]
// CHECK-SAME: ["tune-cpu", "skylake-avx512"]
// CHECK-NEXT: llvm.return %[[ARG0]] : i32
kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK: llvm.func internal @none_type() -> !llvm.struct<()>
kgen.func @none_type() -> !kgen.none {
  // CHECK: [[NONE:%.*]] = llvm.mlir.undef : !llvm.struct<()>
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: llvm.func internal @convert_pop_types
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: %{{.*}}: !llvm.ptr
// CHECK-SAME: %{{.*}}: vector<4xf32>

kgen.func @convert_pop_types(
    %arg0: !pop.simd<1, f32>,
    %arg1: !kgen.pointer<simd<1, f32>>,
    %arg2: !pop.simd<4, f32>) {
  kgen.return
}

kgen.func @trivial_simd(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  kgen.return %arg0 : !pop.simd<1, f32>
}

kgen.func @no_result(%arg0: !pop.simd<1, f32>) {
  kgen.return
}

kgen.func @two_results(%arg0: !pop.simd<1, f32>) -> (!pop.simd<1, f32>, !pop.simd<1, f32>) {
  kgen.return %arg0, %arg0 : !pop.simd<1, f32>, !pop.simd<1, f32>
}

// CHECK-LABEL: llvm.func internal @convert_call
// CHECK-SAME: %[[ARG0:.*]]: f32
kgen.func @convert_call(%arg0: !pop.simd<1, f32>) {
  // CHECK: llvm.call @trivial_simd(%[[ARG0]]) {fastmathFlags = #llvm.fastmath<contract>} : (f32) -> f32
  %0 = kgen.call @trivial_simd(%arg0) : (!pop.simd<1, f32>) -> !pop.simd<1, f32>
  // CHECK: llvm.call @no_result(%[[ARG0]]) {fastmathFlags = #llvm.fastmath<contract>} : (f32) -> ()
  kgen.call @no_result(%arg0) : (!pop.simd<1, f32>) -> ()
  // CHECK: %[[PACK:.*]] = llvm.call @two_results(%[[ARG0]]) {fastmathFlags = #llvm.fastmath<contract>} : (f32) -> !llvm.struct<(f32, f32)>
  %1:2 = kgen.call @two_results(%arg0) : (!pop.simd<1, f32>) -> (!pop.simd<1, f32>, !pop.simd<1, f32>)
  // CHECK: llvm.extractvalue %[[PACK]][0]
  // CHECK: llvm.extractvalue %[[PACK]][1]
  kgen.return
}

//CHECK: "noinline"
kgen.func @test_no_inline(%a: i64) no_inline {
  kgen.return
}

//CHECK: "alwaysinline"
kgen.func @test_always_inline(%a: i64) always_inline {
  kgen.return
}

kgen.func @reference_me(%a: i64) -> i64 {
  kgen.return %a : i64
}

// CHECK-LABEL: @address_dtype
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr
// CHECK-SAME: %[[ARG1:.*]]: !llvm.vec<4 x ptr>
kgen.func @address_dtype(%arg0 : !pop.simd<1, address>, %arg1 : !pop.simd<4, address>) {
  kgen.return
}

// CHECK-LABEL: @unknown
kgen.func @unknown() -> index {
  // CHECK-NEXT: llvm.mlir.undef : i64
  %0 = kgen.param.constant = <*?>
  kgen.return %0 : index
}

kgen.func @constant_str() -> !kgen.string {
  // CHECK: %[[LENGTH:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  // CHECK: %[[GLOBAL_STR:.*]] = llvm.mlir.addressof @[[STATIC_STRING:.*]] : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.bitcast %[[GLOBAL_STR]] : !llvm.ptr to !llvm.ptr
  // CHECK: %[[VAL0:.*]] = llvm.insertvalue %[[GEP]], %[[STRUCT]][0] : !llvm.struct<(ptr, i64)>
  // CHECK: %[[VAL1:.*]] = llvm.insertvalue %[[LENGTH]], %[[VAL0]][1] : !llvm.struct<(ptr, i64)>
  %0 = kgen.param.constant: string = <"AB">
  // CHECK: llvm.return %[[VAL1]] : !llvm.struct<(ptr, i64)>
  kgen.return %0 : !kgen.string
}

kgen.func @constant_str_2() -> !kgen.string {
  // CHECK: llvm.mlir.addressof @[[STATIC_STRING]] : !llvm.ptr
  %0 = kgen.param.constant: string = <"AB">
  kgen.return %0 : !kgen.string
}

// CHECK-LABEL: @undef_op
kgen.func @undef_op() -> i32 {
  // CHECK-NEXT: %0 = llvm.mlir.undef : i32
  // CHECK-NEXT: llvm.return %0 : i32
  %0 = kgen.undef : i32
  kgen.return %0 : i32
}

// CHECK-LABEL: @variant_constant_0
kgen.func @variant_constant_0() -> !kgen.variant<i32> {
  // CHECK: %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %1 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %2 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %3 = llvm.lshr %0, %2  : i32
  // CHECK: %4 = llvm.zext %3 : i32 to i64
  // CHECK: %5 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %6 = llvm.shl %4, %5  : i64
  // CHECK: %7 = llvm.or %1, %6  : i64
  // CHECK: %8 = llvm.mlir.undef : !llvm.array<1 x i64>
  // CHECK: %9 = llvm.insertvalue %7, %8[0] : !llvm.array<1 x i64>
  // CHECK: %10 = llvm.mlir.undef : !llvm.struct<(array<1 x i64>, i8)>
  // CHECK: %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(array<1 x i64>, i8)>
  // CHECK: %12 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %13 = llvm.insertvalue %12, %11[1] : !llvm.struct<(array<1 x i64>, i8)>
  // CHECK: llvm.return %13 : !llvm.struct<(array<1 x i64>, i8)>
  %0 = kgen.param.constant: variant<i32> = <#kgen.variant<:i32 1, 0>>
  kgen.return %0 : !kgen.variant<i32>
}

// CHECK-LABEL: @variant_constant_1
kgen.func @variant_constant_1() -> !kgen.variant<struct<(i32, i64, i32)>, struct<(f64, f32)>> {
  // CHECK: %0 = llvm.mlir.undef : !llvm.struct<(i32, i64, i32)>
  // CHECK: %1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %2 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %3 = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %4 = llvm.insertvalue %3, %2[1] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %5 = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %6 = llvm.insertvalue %5, %4[2] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %7 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %8 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %9 = llvm.extractvalue %6[0] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %10 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %11 = llvm.lshr %9, %10  : i32
  // CHECK: %12 = llvm.zext %11 : i32 to i64
  // CHECK: %13 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %14 = llvm.shl %12, %13  : i64
  // CHECK: %15 = llvm.or %7, %14  : i64
  // CHECK: %16 = llvm.extractvalue %6[1] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %17 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %18 = llvm.lshr %16, %17  : i64
  // CHECK: %19 = llvm.trunc %18 : i64 to i64
  // CHECK: %20 = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %21 = llvm.shl %19, %20  : i64
  // CHECK: %22 = llvm.or %15, %21  : i64
  // CHECK: %23 = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %24 = llvm.lshr %16, %23  : i64
  // CHECK: %25 = llvm.trunc %24 : i64 to i64
  // CHECK: %26 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %27 = llvm.shl %25, %26  : i64
  // CHECK: %28 = llvm.or %8, %27  : i64
  // CHECK: %29 = llvm.extractvalue %6[2] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %30 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %31 = llvm.lshr %29, %30  : i32
  // CHECK: %32 = llvm.zext %31 : i32 to i64
  // CHECK: %33 = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %34 = llvm.shl %32, %33  : i64
  // CHECK: %35 = llvm.or %28, %34  : i64
  // CHECK: %36 = llvm.mlir.undef : !llvm.array<2 x i64>
  // CHECK: %37 = llvm.insertvalue %22, %36[0] : !llvm.array<2 x i64>
  // CHECK: %38 = llvm.insertvalue %35, %37[1] : !llvm.array<2 x i64>
  // CHECK: %39 = llvm.mlir.undef : !llvm.struct<(array<2 x i64>, i8)>
  // CHECK: %40 = llvm.insertvalue %38, %39[0] : !llvm.struct<(array<2 x i64>, i8)>
  // CHECK: %41 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %42 = llvm.insertvalue %41, %40[1] : !llvm.struct<(array<2 x i64>, i8)>
  // CHECK: lvm.return %42 : !llvm.struct<(array<2 x i64>, i8)>
  %0 = kgen.param.constant: variant<struct<(i32, i64, i32)>, struct<(f64, f32)>> = <#kgen.variant<:!kgen.struct<(i32, i64, i32)> { 1, 2, 3 }, 0>>
  kgen.return %0 : !kgen.variant<struct<(i32, i64, i32)>, struct<(f64, f32)>>
}

// CHECK-LABEL: @variant_constant_2
kgen.func @variant_constant_2() -> !kgen.variant<i1, i2, i3, i4, i5, i6> {
  // CHECK: %0 = llvm.mlir.constant(1 : i4) : i4
  // CHECK: %1 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %2 = llvm.mlir.constant(0 : i4) : i4
  // CHECK: %3 = llvm.lshr %0, %2  : i4
  // CHECK: %4 = llvm.zext %3 : i4 to i64
  // CHECK: %5 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %6 = llvm.shl %4, %5  : i64
  // CHECK: %7 = llvm.or %1, %6  : i64
  // CHECK: %8 = llvm.mlir.undef : !llvm.array<1 x i64>
  // CHECK: %9 = llvm.insertvalue %7, %8[0] : !llvm.array<1 x i64>
  // CHECK: %10 = llvm.mlir.undef : !llvm.struct<(array<1 x i64>, i8)>
  // CHECK: %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(array<1 x i64>, i8)>
  // CHECK: %12 = llvm.mlir.constant(3 : i8) : i8
  // CHECK: %13 = llvm.insertvalue %12, %11[1] : !llvm.struct<(array<1 x i64>, i8)>
  // CHECK: llvm.return %13 : !llvm.struct<(array<1 x i64>, i8)>
  %0 = kgen.param.constant: variant<i1, i2, i3, i4, i5, i6> = <#kgen.variant<:i4 1, 3>>
  kgen.return %0 : !kgen.variant<i1, i2, i3, i4, i5, i6>
}

// CHECK-LABEL: @test_unreachable
kgen.func @test_unreachable() -> !pop.simd<1, f32> {
  // CHECK-NEXT: llvm.trap
  // CHECK-NEXT: llvm.unreachable
  kgen.unreachable
}

// CHECK-LABEL: @address_of
kgen.func @address_of() -> !kgen.signature<() -> !pop.scalar<f32>> {
  // CHECK: llvm.mlir.addressof @test_unreachable : !llvm.ptr
  %0 = kgen.param.constant: () -> !pop.scalar<f32> = <@test_unreachable>
  kgen.return %0 : !kgen.signature<() -> !pop.scalar<f32>>
}

// CHECK: llvm.func internal @used_internally_c_wrapped
kgen.func export C @used_internally() -> !kgen.struct<(i32, i32)>{
  kgen.unreachable
}

// CHECK: llvm.func @used_internally

// CHECK: llvm.func internal @used_func
kgen.func @used_func() {
  // CHECK-NEXT: call @used_internally_c_wrapped
  kgen.call @used_internally() : () -> !kgen.struct<(i32, i32)>
  kgen.return
}

// CHECK: llvm.func @used_package_func
kgen.func export package @used_package_func() -> !kgen.struct<(i32, i32)>{
  kgen.unreachable
}

// CHECK: llvm.mlir.global internal constant @[[STATIC_STRING]]("AB\00") {addr_space = 0 : i32, alignment = 16 : i64}

}

// -----

// CHECK-LABEL: module
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // CHECK: llvm.mlir.global_ctors {ctors = [@kgenGlobalCtor], priorities = [0 : i32]}
  // CHECK: llvm.mlir.global_dtors {dtors = [@kgenGlobalDtor], priorities = [0 : i32]}

  // CHECK: llvm.func weak @kgenGlobalCtor
  // CHECK-NEXT: call @noop()
  // CHECK-NEXT: call @foo_c()
  // CHECK-NEXT: call @bar_c()

  // CHECK: llvm.func weak @kgenGlobalDtor
  // CHECK-NEXT: call @bar_d()
  // CHECK-NEXT: call @foo_d()
  // CHECK-NEXT: call @noop()

  llvm.func @foo_c() {
    llvm.return
  }
  llvm.func @foo_d() {
    llvm.return
  }
  llvm.func @bar_c() {
    llvm.return
  }
  llvm.func @bar_d() {
    llvm.return
  }
  llvm.func @noop() {
    llvm.return
  }

  kgen.global @foo : i32 [@foo_c, @foo_d](2)
  kgen.global @bar : i64 [@bar_c, @bar_d](5)
  kgen.global @exported : f32 [@noop, @noop](0)
}

// -----

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
// CHECK-LABEL: llvm.func @kernel() attributes {dso_local, nvvm.kernel
kgen.func export @kernel() {
  kgen.return
}
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @struct_constant
kgen.func @struct_constant() -> !kgen.struct<(array<1, i32>, struct<(i32, i32)>)> {
  // CHECK: %0 = llvm.mlir.undef : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  // CHECK: %1 = llvm.mlir.undef : !llvm.array<1 x i32>
  // CHECK: %2 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %3 = llvm.insertvalue %2, %1[0] : !llvm.array<1 x i32>
  // CHECK: %4 = llvm.insertvalue %3, %0[0] : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  // CHECK: %5 = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK: %6 = llvm.mlir.constant(2 : i32) : i32
  // CHECK: %7 = llvm.insertvalue %6, %5[0] : !llvm.struct<(i32, i32)>
  // CHECK: %8 = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %9 = llvm.insertvalue %8, %7[1] : !llvm.struct<(i32, i32)>
  // CHECK: %10 = llvm.insertvalue %9, %4[1] : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  // CHECK: llvm.return %10 : !llvm.struct<(array<1 x i32>, struct<(i32, i32)>)>
  %0 = kgen.param.constant: struct<(array<1, i32>, struct<(i32, i32)>)> =
    <{ [1], { 2, 3 } }>
  kgen.return %0 : !kgen.struct<(array<1, i32>, struct<(i32, i32)>)>
}

// CHECK-LABEL: @pointer_constant
kgen.func @pointer_constant() -> !kgen.pointer<*?> {
  // CHECK: %0 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %1 = llvm.inttoptr %0 : i64 to !llvm.ptr
  // CHECK: llvm.return %1 : !llvm.ptr
  %null = kgen.param.constant: pointer<*?> = <#interp.pointer<0>>
  kgen.return %null : !kgen.pointer<*?>
}

// CHECK-LABEL: @test_variant
kgen.func @test_variant(%a: !kgen.variant<f32, i64, struct<(i8, i8, f64)>>) -> i1 {
  // CHECK: %[[DISCR:.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(array<2 x i64>, i8)>
  // CHECK: %[[DISCR_VAL:.*]] = llvm.mlir.constant(0 : i8)
  // CHECK: %[[VAL:.*]] = llvm.icmp "eq" %[[DISCR]], %[[DISCR_VAL]]
  %0 = kgen.variant.is %a, 0 : <f32, i64, struct<(i8, i8, f64)>>
  // CHECK: return %[[VAL]]
  kgen.return %0 : i1
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {

// CHECK-LABEL: llvm.func @llvm_metadata
// CHECK-SAME: nvvm.intval = 4 : i64
// CHECK-SAME: nvvm.maxntid = array<i32: 256, 1, 4>
// CHECK-SAME: nvvm.unitattr
// CHECK-SAME: passthrough = [{{.*}}, ["intval", "2"], ["strval", "hello"], "unitattr"]
kgen.func export @llvm_metadata() attributes {
  LLVMMetadata = {
    llvm.unitattr,
    llvm.intval = 2,
    llvm.strval = "hello",

    nvvm.unitattr,
    nvvm.intval = 4,
    nvvm.maxntid = #pop.array<256, 1, 4> : !pop.array<3, i32>
  }
} {
  kgen.return
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {

// CHECK-LABEL: llvm.func internal @coro
// CHECK-SAME: coroutineType = !llvm.struct<(i64, ptr, ptr, ptr, ptr, ptr, ptr)>
kgen.func @coro() attributes {coroutineType = !kgen.struct<(index, (!kgen.pointer<none>) -> (), (!kgen.pointer<none>) -> !kgen.none, pointer<none>, pointer<none>, pointer<none>, pointer<none>)>} {
  kgen.return
}

}
