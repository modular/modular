// RUN: kgen-opt -lower-kgen-to-llvm --split-input-file %s | FileCheck %s
//
// Tests for the C ABI classification of KGEN types.
//
// TRANSFORM UNDER TEST:
//   Mojo/lib/KGENToLLVM/CABILowering.cpp - forEachMember for KGEN types
//   Mojo/lib/KGENToLLVM/CABISystemV.cpp  - classifyStructType
//   Mojo/lib/KGENToLLVM/CABIAAPCS.cpp    - classifyHomogeneousAggregate
//
// The classification walks the KGEN type, not its LLVM type. The LLVM type of
// a union holds only one member, and the LLVM type of a struct holds padding
// that looks like a byte array.

module attributes {M.target_info = #M.target<triple="x86_64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", simd_bit_width=128>} {

// The f64 and the si32 overlap, so the eightbyte is INTEGER.
// clang: union { double; int; } is (i64).
// CHECK-LABEL: llvm.func internal @x86_union_float_int
// CHECK-SAME:    (%{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @x86_union_float_int(%u: !pop.union<scalar<f64>, scalar<si32>>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// The order of the union members does not change the classification.
// clang: struct { union { long; double; }; char; } is (i64, i64).
// CHECK-LABEL: llvm.func internal @x86_tagged_int_float
// CHECK-SAME:    (%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @x86_tagged_int_float(%v: !kgen.struct<(union<scalar<si64>, scalar<f64>>, scalar<ui8>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// CHECK-LABEL: llvm.func internal @x86_tagged_float_int
// CHECK-SAME:    (%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @x86_tagged_float_int(%v: !kgen.struct<(union<scalar<f64>, scalar<si64>>, scalar<ui8>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// An untagged union also classifies all of its members, as an enum does.
// clang: struct { union { long; double; }; char; } is (i64, i64).
// CHECK-LABEL: llvm.func internal @x86_untagged_int_float
// CHECK-SAME:    (%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @x86_untagged_int_float(%v: !kgen.struct<(untagged_union<scalar<si64>, scalar<f64>>, scalar<ui8>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// A union of floats is SSE. clang: union { double; double; } is (double).
// CHECK-LABEL: llvm.func internal @x86_union_floats
// CHECK-SAME:    (%{{.*}}: f64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @x86_union_floats(%u: !pop.union<scalar<f64>, scalar<f64>>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// The tail padding after the f32 is not INTEGER data.
// clang: struct { float __attribute__((vector_size(8))); float; } is
// (double, float). The second part is f64 here; it uses the same SSE register.
// CHECK-LABEL: llvm.func internal @x86_tail_padding
// CHECK-SAME:    (%{{.*}}: f64, %{{.*}}: f64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @x86_tail_padding(%s: !kgen.struct<(simd<2, f32>, scalar<f32>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// A union result. clang: union { double; int; } is returned in i64.
// CHECK-LABEL: llvm.func internal @x86_ret_union
// CHECK-SAME:    (%{{.*}}: i64) -> i64
kgen.func @x86_ret_union(%u: !pop.union<scalar<f64>, scalar<si32>>) cabi -> !pop.union<scalar<f64>, scalar<si32>> {
  hlcf.return %u : !pop.union<scalar<f64>, scalar<si32>>
}

// The upper eightbyte holds only padding, so it is NO_CLASS and is not passed.
// The pair after it then needs two registers but only one is free, so it goes
// in memory and the last i64 takes the register.
// clang: struct __attribute__((aligned(16))) { long; } is (i64).
// CHECK-LABEL: llvm.func internal @x86_align16_int
// CHECK-SAME:    (%{{.*}}: i64 {llvm.noundef}, %{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}, %{{.*}}: i64 {llvm.noundef}, %{{.*}}: i64 {llvm.noundef}, %{{.*}}: !llvm.ptr {llvm.byval = !llvm.struct<(i64, i64)>{{.*}}}, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @x86_align16_int(%a: i64, %s: !kgen.struct<(scalar<si64>) align(16)>, %b: i64, %c: i64, %d: i64, %p: !kgen.struct<(scalar<si64>, scalar<si64>)>, %t: i64) cabi -> i64 {
  hlcf.return %t : i64
}

// clang: struct __attribute__((aligned(16))) { double; } is (double).
// CHECK-LABEL: llvm.func internal @x86_align16_float
// CHECK-SAME:    (%{{.*}}: f64) -> f64
kgen.func @x86_align16_float(%s: !kgen.struct<(scalar<f64>) align(16)>) cabi -> !kgen.struct<(scalar<f64>) align(16)> {
  hlcf.return %s : !kgen.struct<(scalar<f64>) align(16)>
}

}

// -----

module attributes {M.target_info = #M.target<triple="aarch64-unknown-linux-gnu", arch="", features="", data_layout="e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128", simd_bit_width=128>} {

// The si64 member makes the union not homogeneous, so it goes in a GPR.
// clang: union { double; long; } is (i64).
// CHECK-LABEL: llvm.func internal @arm_union_float_int
// CHECK-SAME:    (%{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_union_float_int(%u: !pop.union<scalar<f64>, scalar<si64>>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// The members of a union overlap, so they count once.
// clang: union { double; double; } is an HFA of one double.
// CHECK-LABEL: llvm.func internal @arm_union_floats
// CHECK-SAME:    (%{{.*}}: !llvm.array<1 x f64>, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_union_floats(%u: !pop.union<scalar<f64>, scalar<f64>>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// The members of an untagged union overlap too. The largest member gives the
// number of floats. clang: union { float; struct { float a, b; }; } is
// [2 x float].
// CHECK-LABEL: llvm.func internal @arm_untagged_floats
// CHECK-SAME:    (%{{.*}}: !llvm.array<2 x f32>, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_untagged_floats(%u: !kgen.struct<(untagged_union<scalar<f32>, struct<(scalar<f32>, scalar<f32>)>>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// An empty member holds no data, so it does not count. clang: [1 x double].
// CHECK-LABEL: llvm.func internal @arm_lead_empty
// CHECK-SAME:    (%{{.*}}: !llvm.array<1 x f64>, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_lead_empty(%s: !kgen.struct<(struct<()>, scalar<f64>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// Five floats are too many members for an HFA, so the 20-byte struct goes in
// memory. clang: ptr.
// CHECK-LABEL: llvm.func internal @arm_nested_five
// CHECK-SAME:    (%{{.*}}: !llvm.ptr {llvm.noundef}, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_nested_five(%s: !kgen.struct<(struct<(scalar<f32>, scalar<f32>, scalar<f32>)>, struct<(scalar<f32>, scalar<f32>)>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// A short vector and a float are different base types. clang: [2 x i64].
// CHECK-LABEL: llvm.func internal @arm_vector_and_float
// CHECK-SAME:    (%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_vector_and_float(%s: !kgen.struct<(simd<2, f32>, scalar<f32>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// An HFA has no padding. clang: i128.
// CHECK-LABEL: llvm.func internal @arm_padded_float
// CHECK-SAME:    (%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_padded_float(%s: !kgen.struct<(scalar<f32>) align(16)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// Vectors of the same 128-bit size are an HVA, whatever their element type.
// clang: struct { int32x4_t a, b; } is [2 x <4 x i32>].
// CHECK-LABEL: llvm.func internal @arm_int_vectors
// CHECK-SAME:    (%{{.*}}: !llvm.array<2 x vector<4xi32>>, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_int_vectors(%s: !kgen.struct<(simd<4, si32>, simd<4, si32>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

// A 256-bit vector is not a base type, so the 32-byte struct goes in memory.
// clang: struct { double v __attribute__((vector_size(32))); } is ptr.
// CHECK-LABEL: llvm.func internal @arm_wide_vector
// CHECK-SAME:    (%{{.*}}: !llvm.ptr {llvm.noundef}, %{{.*}}: i64 {llvm.noundef}) -> i64
kgen.func @arm_wide_vector(%s: !kgen.struct<(simd<4, f64>)>, %x: i64) cabi -> i64 {
  hlcf.return %x : i64
}

}
