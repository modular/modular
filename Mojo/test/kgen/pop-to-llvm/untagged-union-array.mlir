// Check compact and padded union conversion in SSA array create, repeat, get and replace operations.
// RUN: kgen-opt %s -pass-pipeline='builtin.module(kgen-verifier,lower-kgen-to-llvm,lower-global-pop-to-llvm,llvm.func(lower-pop-to-llvm,lower-control-flow,reconcile-unrealized-casts))' | kgen-translate -mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define internal { i64, [32 x i8] } @array_ssa(
// CHECK: insertvalue { { i64, [32 x i8] }, [24 x i8] } undef, { i64, [32 x i8] }
// CHECK: insertvalue [2 x { { i64, [32 x i8] }, [24 x i8] }]
// CHECK: insertvalue { { i64, [32 x i8] }, [24 x i8] } undef, { i64, [32 x i8] }
// CHECK: insertvalue [2 x { { i64, [32 x i8] }, [24 x i8] }]
// CHECK: extractvalue [2 x { { i64, [32 x i8] }, [24 x i8] }]
// CHECK: extractvalue { { i64, [32 x i8] }, [24 x i8] } {{.*}}, 0
// CHECK: insertvalue { { i64, [32 x i8] }, [24 x i8] } undef, { i64, [32 x i8] }
// CHECK: insertvalue [2 x { { i64, [32 x i8] }, [24 x i8] }]
// CHECK: extractvalue [2 x { { i64, [32 x i8] }, [24 x i8] }]
// CHECK: extractvalue { { i64, [32 x i8] }, [24 x i8] } {{.*}}, 0
// CHECK: ret { i64, [32 x i8] }

module attributes {M.target_info = #M.target<triple="arm64-apple-darwin25.6.0", arch="", features="", data_layout="e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32", simd_bit_width=128>} {
kgen.func @array_ssa(%x: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>, %y: !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>) -> !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>> {
 %a = pop.array.create [%x, %y] : !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 %b = pop.array.repeat [%y] : !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 %c = pop.array.get %a[1] : !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 %d = pop.array.replace %c, %b[0] : !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 %e = pop.array.get %d[0] : !pop.array<2, !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>>
 hlcf.return %e : !kgen.untagged_union<simd<4, si64>, struct<(i64, i64, i64, i64, i64)>>
}
}
