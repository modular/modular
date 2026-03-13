// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// #pop.simd_cmp with index: target-dependent folding on 32-bit.
// 3000000000 wraps to negative in 32-bit signed, so lt(3000000000, 0) is true.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_cmp_index_32
kgen.generator export @test_simd_cmp_index_32() -> !pop.scalar<bool> {
  kgen.param.declare value : !pop.scalar<bool> = <#pop.simd_cmp<lt, #pop<simd 3000000000> : !pop.scalar<index>, #pop<simd 0> : !pop.scalar<index>> : !pop.scalar<bool>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !pop.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !pop.scalar<bool>
  kgen.return %0 : !pop.scalar<bool>
}
}

// -----

// #pop.simd_cmp with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so lt(3000000000, 0) is false.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_cmp_index_64
kgen.generator export @test_simd_cmp_index_64() -> !pop.scalar<bool> {
  kgen.param.declare value : !pop.scalar<bool> = <#pop.simd_cmp<lt, #pop<simd 3000000000> : !pop.scalar<index>, #pop<simd 0> : !pop.scalar<index>> : !pop.scalar<bool>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !pop.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !pop.scalar<bool>
  kgen.return %0 : !pop.scalar<bool>
}
}

// -----

// #pop.simd_shl with index: target-dependent folding on 64-bit.
// shl(1, 33) is poison on 32-bit (shift >= 32), so it can't fold without
// target. With 64-bit target, shl(1, 33) = 8589934592.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shl_index_64
kgen.generator export @test_simd_shl_index_64() -> !pop.scalar<index> {
  kgen.param.declare value : !pop.scalar<index> = <#pop.simd_shl<#pop<simd 1> : !pop.scalar<index>, #pop<simd 33> : !pop.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <8589934592>
  %0 = kgen.param.constant: !pop.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}
}

// -----

// #pop.simd_shr with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed, so ashr(-1294967296, 1) = -647483648.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shr_index_32
kgen.generator export @test_simd_shr_index_32() -> !pop.scalar<index> {
  kgen.param.declare value : !pop.scalar<index> = <#pop.simd_shr<#pop<simd 3000000000> : !pop.scalar<index>, #pop<simd 1> : !pop.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <-647483648>
  %0 = kgen.param.constant: !pop.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}
}

// -----

// #pop.simd_shr with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so shr(3000000000, 1) = 1500000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shr_index_64
kgen.generator export @test_simd_shr_index_64() -> !pop.scalar<index> {
  kgen.param.declare value : !pop.scalar<index> = <#pop.simd_shr<#pop<simd 3000000000> : !pop.scalar<index>, #pop<simd 1> : !pop.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1500000000>
  %0 = kgen.param.constant: !pop.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}
}

// -----

// #pop.simd_abs with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed, so abs(-1294967296) = 1294967296.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_abs_index_32
kgen.generator export @test_simd_abs_index_32() -> !pop.scalar<index> {
  kgen.param.declare value : !pop.scalar<index> = <#pop.simd_abs<#pop<simd 3000000000> : !pop.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1294967296>
  %0 = kgen.param.constant: !pop.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}
}

// -----

// #pop.simd_abs with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so abs(3000000000) = 3000000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_abs_index_64
kgen.generator export @test_simd_abs_index_64() -> !pop.scalar<index> {
  kgen.param.declare value : !pop.scalar<index> = <#pop.simd_abs<#pop<simd 3000000000> : !pop.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <3000000000>
  %0 = kgen.param.constant: !pop.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !pop.scalar<index>
  kgen.return %0 : !pop.scalar<index>
}
}
