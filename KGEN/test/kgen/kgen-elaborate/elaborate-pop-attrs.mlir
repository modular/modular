// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// #pop.simd_cmp with index: target-dependent folding on 32-bit.
// 3000000000 wraps to negative in 32-bit signed, so lt(3000000000, 0) is true.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_cmp_index_32
kgen.generator export @test_simd_cmp_index_32() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#pop.simd_cmp<lt, #kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>> : !kgen.scalar<bool>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <true>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// #pop.simd_cmp with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so lt(3000000000, 0) is false.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_cmp_index_64
kgen.generator export @test_simd_cmp_index_64() -> !kgen.scalar<bool> {
  kgen.param.declare value : !kgen.scalar<bool> = <#pop.simd_cmp<lt, #kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 0> : !kgen.scalar<index>> : !kgen.scalar<bool>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<bool> = <false>
  %0 = kgen.param.constant: !kgen.scalar<bool> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<bool>
  kgen.return %0 : !kgen.scalar<bool>
}
}

// -----

// #pop.simd_shl with index: target-dependent folding on 64-bit.
// shl(1, 33) is poison on 32-bit (shift >= 32), so it can't fold without
// target. With 64-bit target, shl(1, 33) = 8589934592.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shl_index_64
kgen.generator export @test_simd_shl_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_shl<#kgen<simd 1> : !kgen.scalar<index>, #kgen<simd 33> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <8589934592>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_shr with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed, so ashr(-1294967296, 1) = -647483648.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shr_index_32
kgen.generator export @test_simd_shr_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_shr<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 1> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <-647483648>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_shr with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so shr(3000000000, 1) = 1500000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_shr_index_64
kgen.generator export @test_simd_shr_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_shr<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 1> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1500000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_abs with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed, so abs(-1294967296) = 1294967296.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_abs_index_32
kgen.generator export @test_simd_abs_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_abs<#kgen<simd 3000000000> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1294967296>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_abs with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so abs(3000000000) = 3000000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_abs_index_64
kgen.generator export @test_simd_abs_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_abs<#kgen<simd 3000000000> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <3000000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_div with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed,
// so div(-1294967296, 2) = -647483648.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_div_index_32
kgen.generator export @test_simd_div_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_div<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <-647483648>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_div with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so div(3000000000, 2) = 1500000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_div_index_64
kgen.generator export @test_simd_div_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_div<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1500000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_floordiv with index: target-dependent folding on 32-bit.
// 3000000000 wraps to -1294967296 in 32-bit signed,
// so floordiv(-1294967296, 2) = -647483648.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_floordiv_index_32
kgen.generator export @test_simd_floordiv_index_32() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_floordiv<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <-647483648>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}

// -----

// #pop.simd_floordiv with index: target-dependent folding on 64-bit.
// 3000000000 fits in 64-bit as positive, so floordiv(3000000000, 2) = 1500000000.

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "", simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
// CHECK-LABEL: kgen.func export @test_simd_floordiv_index_64
kgen.generator export @test_simd_floordiv_index_64() -> !kgen.scalar<index> {
  kgen.param.declare value : !kgen.scalar<index> = <#pop.simd_floordiv<#kgen<simd 3000000000> : !kgen.scalar<index>, #kgen<simd 2> : !kgen.scalar<index>>>
  // CHECK-NEXT: [[V0:%.*]] = kgen.param.constant: scalar<index> = <1500000000>
  %0 = kgen.param.constant: !kgen.scalar<index> = <value>
  // CHECK-NEXT: kgen.return [[V0]] : !kgen.scalar<index>
  kgen.return %0 : !kgen.scalar<index>
}
}
