// RUN: kgen-opt -lower-kgen-to-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="skylake-avx512", features="+fma", data_layout="", simd_bit_width=128, tune_cpu="skylake-avx512">} {

// CHECK-LABEL: llvm.func internal @trivial
// CHECK-SAME: (%[[ARG0:.*]]: i32)
// CHECK-SAME: ["target-cpu", "skylake-avx512"]
// CHECK-SAME: ["target-features", "+fma"]
// CHECK-SAME: ["tune-cpu", "skylake-avx512"]
// CHECK-NEXT: llvm.return %[[ARG0]] : i32
kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: llvm.func internal @convert_pop_types
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: %{{.*}}: !llvm.ptr<f32>
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

kgen.func @reference_me(%a: i64) -> i64 {
  kgen.return %a : i64
}

// CHECK-LABEL: @address_dtype
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME: %[[ARG1:.*]]: !llvm.vec<4 x ptr>
kgen.func @address_dtype(%arg0 : !pop.simd<1, address>, %arg1 : !pop.simd<4, address>) {
  kgen.return
}

kgen.func @constant_str() -> !kgen.string {
  // CHECK: %[[LENGTH:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
  // CHECK: %[[GLOBAL_STR:.*]] = llvm.mlir.addressof @[[STATIC_STRING:.*]] : !llvm.ptr<array<3 x i8>>
  // CHECK: %[[GEP:.*]] = llvm.bitcast %[[GLOBAL_STR]] : !llvm.ptr<array<3 x i8>> to !llvm.ptr<i8>
  // CHECK: %[[VAL0:.*]] = llvm.insertvalue %[[GEP]], %[[STRUCT]][0] : !llvm.struct<(ptr<i8>, i64)>
  // CHECK: %[[VAL1:.*]] = llvm.insertvalue %[[LENGTH]], %[[VAL0]][1] : !llvm.struct<(ptr<i8>, i64)>
  %0 = kgen.param.constant: string = <"AB">
  // CHECK: llvm.return %[[VAL1]] : !llvm.struct<(ptr<i8>, i64)>
  kgen.return %0 : !kgen.string
}

kgen.func @constant_str_2() -> !kgen.string {
  // CHECK: llvm.mlir.addressof @[[STATIC_STRING]] : !llvm.ptr<array<3 x i8>>
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

// CHECK-LABEL: @pack_constant
kgen.func @pack_constant() {
  // CHECK-NEXT: %0 = llvm.mlir.undef : !llvm.struct<(i64)>
  // CHECK-NEXT: %1 = llvm.mlir.constant(1 : i64)
  // CHECK-NEXT: %2 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i64)>
  %0 = kgen.param.constant: !pop.pack<[!pop.scalar<index>]> = <<1>>
  kgen.return
}

// CHECK-LABEL: @test_unreachable
kgen.func @test_unreachable() -> !pop.simd<1, f32> {
  // CHECK-NEXT: llvm.trap
  // CHECK-NEXT: llvm.unreachable
  kgen.unreachable
}

// CHECK: llvm.func internal @used_internally_c_wrapped
kgen.func export C @used_internally() -> !pop.struct<i32, i32>{
  kgen.unreachable
}

// CHECK: llvm.func @used_internally

// CHECK: llvm.func internal @used_func
kgen.func @used_func() {
  // CHECK-NEXT: call @used_internally_c_wrapped
  kgen.call @used_internally() : () -> !pop.struct<i32, i32>
  kgen.return
}

// CHECK: llvm.func extern_weak @external_func()
kgen.link "/path/to/libc.a" as @libc
kgen.extern.func @external_func() -> () from @libc

// CHECK: llvm.mlir.global internal constant @[[STATIC_STRING]]("AB\00") {addr_space = 0 : i32, alignment = 16 : i64}

}
