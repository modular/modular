// 1. Allocate 40-byte unions at alignment 32 with a 64-byte stride.
// 2. Load and store only 40 value bytes, preserving 24 bytes of tail padding.
// 3. Lower aligned enum fields and relocatable global and function constants.
// RUN: kgen-opt %s -kgen-verifier -lower-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

// CHECK-DAG: @raw_union_two = internal global [2 x { { i64, [32 x i8] }, [24 x i8] }] undef, align 32
// CHECK-DAG: @raw_union_array = internal global [1 x [2 x { { i64, [32 x i8] }, [24 x i8] }]] undef, align 32
// CHECK-DAG: @global_constant = internal constant <{ i8, {{.*}}[24 x i8] }> {{.*}}, align 32
// CHECK-DAG: @global_constant_0 = internal constant <{ i8, {{.*}}[24 x i8], i8, {{.*}}[24 x i8] }> {{.*}}, align 32
// CHECK-DAG: @raw_union_init = internal global [1 x <{ i8, {{.*}}[24 x i8] }>] {{.*}}, align 32
// CHECK-DAG: @raw_union_symbol = internal global [1 x <{ {{.*}}ptr }>] {{.*}}ptr @relocated_callback{{.*}}, align 32

module attributes {M.target_info = #M.target<triple="arm64-apple-darwin25.6.0", arch="", features="", data_layout="e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32", simd_bit_width=128>} {

// CHECK-LABEL: define internal void @stack_two(
// CHECK: alloca [2 x { { i64, [32 x i8] }, [24 x i8] }], align 32
// CHECK: store { i64, [32 x i8] } {{.*}}, align 32
kgen.func @stack_two(%value: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>) {
 %p = pop.stack_allocation 2 x !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>
 pop.store %value, %p : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return
}

// CHECK-LABEL: define internal void @array_stack(
// CHECK: alloca [2 x { { i64, [32 x i8] }, [24 x i8] }], i64 1, align 32
// CHECK: getelementptr inbounds [2 x { { i64, [32 x i8] }, [24 x i8] }], {{.*}}, i32 0, i32 1
// CHECK: store { i64, [32 x i8] } {{.*}}, align 32
kgen.func @array_stack(%value: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>) {
 %p = pop.stack_allocation 1 x !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 %one = index.constant 1
 %gepidx = pop.cast_from_builtin %one : index to !kgen.scalar<index>
 %q = pop.array.gep %p[%gepidx] : !kgen.pointer<!pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>>
 pop.store %value, %q : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return
}

// CHECK-LABEL: define internal void @global_two(
// CHECK: store { i64, [32 x i8] } {{.*}}i64 64), align 32
kgen.func @global_two(%value: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>) {
 %p = pop.global_alloc "raw_union_two" 2 x !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>
 %one = index.constant 1
 %offidx = pop.cast_from_builtin %one : index to !kgen.scalar<index>
 %q = pop.offset %p[%offidx] : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 pop.store %value, %q : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return
}

// CHECK-LABEL: define internal void @global_array(
// CHECK: store { i64, [32 x i8] } {{.*}}i64 64), align 32
kgen.func @global_array(%value: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>) {
 %p = pop.global_alloc "raw_union_array" 1 x !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 %one = index.constant 1
 %gepidx = pop.cast_from_builtin %one : index to !kgen.scalar<index>
 %q = pop.array.gep %p[%gepidx] : !kgen.pointer<!pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>>
 pop.store %value, %q : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return
}

// CHECK-LABEL: define internal ptr @offset_one(
// CHECK: getelementptr inbounds { { i64, [32 x i8] }, [24 x i8] }, {{.*}}, i32 1
kgen.func @offset_one(%p: !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>) -> !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>> {
 %one = index.constant 1
 %offidx = pop.cast_from_builtin %one : index to !kgen.scalar<index>
 %q = pop.offset %p[%offidx] : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return %q : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
}

// CHECK-LABEL: define internal ptr @array_gep(
// CHECK: getelementptr inbounds [2 x { { i64, [32 x i8] }, [24 x i8] }], {{.*}}, i32 0, i32 1
kgen.func @array_gep(%p: !kgen.pointer<!pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>>) -> !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>> {
 %one = index.constant 1
 %gepidx = pop.cast_from_builtin %one : index to !kgen.scalar<index>
 %q = pop.array.gep %p[%gepidx] : !kgen.pointer<!pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>>
 hlcf.return %q : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
}

kgen.func @constant_union() {
 %p = pop.global_constant: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>> = <#kgen.untagged_union<[7 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8]> : !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return
}

kgen.func @constant_array() {
 %p = pop.global_constant: !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>> = <[#kgen.untagged_union<[7 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8]> : !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>, #kgen.untagged_union<[7 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8]> : !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>]>
 hlcf.return
}

kgen.func @global_initialized() {
 %p = pop.global_alloc "raw_union_init" 1 x !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>> = <#kgen.untagged_union<[7 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8]> : !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return
}

kgen.func @relocated_callback() {
 hlcf.return
}

kgen.func @global_initialized_symbol() {
 %p = pop.global_alloc "raw_union_symbol" 1 x !kgen.untagged_union<struct<(i64, () -> ())>> align 32 = <#kgen.untagged_union<[5 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, #kgen.func.symbol<@relocated_callback> : !kgen.func<() -> ()>]> : !kgen.untagged_union<struct<(i64, () -> ())>>>
 hlcf.return
}

// CHECK-LABEL: define internal void @heap(
// CHECK: call ptr @KGEN_CompilerRT_AlignedAlloc(i64 32, i64 128)
// CHECK: getelementptr inbounds { { i64, [32 x i8] }, [24 x i8] }, {{.*}}, i32 1
// CHECK: store { i64, [32 x i8] } {{.*}}, align 32
// CHECK: store { i64, [32 x i8] } {{.*}}, align 32
kgen.func @heap(%value: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>) {
 %align = kgen.param.constant: scalar<index> = <32>
 %bytes = kgen.param.constant: scalar<index> = <128>
 %p = pop.aligned_alloc %align, %bytes : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 %one = index.constant 1
 %offidx = pop.cast_from_builtin %one : index to !kgen.scalar<index>
 %q = pop.offset %p[%offidx] : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 pop.store %value, %p : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 pop.store %value, %q : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 pop.aligned_free %p : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return
}

// CHECK-LABEL: define internal ptr @enum_field(
// CHECK: getelementptr { { i64, [32 x i8] }, i8, [23 x i8] }, {{.*}}, i32 0, i32 1
kgen.func @enum_field(%p: !kgen.pointer<struct<(!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>, i8)>>) -> !kgen.pointer<i8> {
 %q = kgen.struct.gep %p[1] : <struct<(!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>, i8)>>
 hlcf.return %q : !kgen.pointer<i8>
}

// CHECK-LABEL: define internal ptr @enum_field_24(
// CHECK: getelementptr { [24 x i8], i8, [7 x i8] }, {{.*}}, i32 0, i32 1
kgen.func @enum_field_24(%p: !kgen.pointer<struct<(!kgen.untagged_union<simd<2, si64>, struct<(i64, i64, i64)>>, i8)>>) -> !kgen.pointer<i8> {
 %q = kgen.struct.gep %p[1] : <struct<(!kgen.untagged_union<simd<2, si64>, struct<(i64, i64, i64)>>, i8)>>
 hlcf.return %q : !kgen.pointer<i8>
}

// CHECK-LABEL: define internal void @enum_stack(
// CHECK: alloca { { i64, [32 x i8] }, i8, [23 x i8] }, i64 1, align 32
kgen.func @enum_stack() {
 %p = pop.stack_allocation 1 x !kgen.struct<(!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>, i8)>
 hlcf.return
}

// CHECK-LABEL: define internal { i64, [32 x i8] } @load_one(
// CHECK: %[[VALUE:.*]] = load { i64, [32 x i8] }, ptr {{.*}}, align 32
// CHECK-NEXT: ret { i64, [32 x i8] } %[[VALUE]]
kgen.func @load_one(%p: !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>) -> !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>> {
 %value = pop.load %p : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return %value : !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>
}

// CHECK-LABEL: define internal ptr @stored_constant(
// CHECK-COUNT-2: alloca { { i64, [32 x i8] }, [24 x i8] }, i64 1, align 32
// CHECK: store i8 7, ptr {{.*}}, align 32
// CHECK: %[[VALUE:.*]] = load { i64, [32 x i8] }, ptr {{.*}}, align 32
// CHECK-NEXT: store { i64, [32 x i8] } %[[VALUE]], ptr %[[SLOT:.*]], align 32
// CHECK-NEXT: ret ptr %[[SLOT]]
kgen.func @stored_constant() -> !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>> {
  %p = kgen.param.constant: pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>> = <store_to_mem(#kgen.untagged_union<[7 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8]> : !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>)>
  hlcf.return %p : !kgen.pointer<!kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
}
}
