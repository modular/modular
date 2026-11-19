// RUN: kgen-opt --split-input-file -kgen-verifier %s --verify-diagnostics

// Non-positive, non-power-of-two SIMD lengths must not reach codegen
module {
  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<3, f32>'}}
  kgen.func @simd_non_pow2(%arg0: !kgen.simd<3, f32>) {
    hlcf.return
  }

  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<0, f32>'}}
  kgen.func @simd_zero(%arg0: !kgen.simd<0, f32>) {
    hlcf.return
  }

  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<-1, f32>'}}
  kgen.func @simd_neg(%arg0: !kgen.simd<-1, f32>) {
    hlcf.return
  }

  // expected-error @below {{SIMD vector length must be a power of two between 1 and 2^15, found '!kgen.simd<65536, f32>'}}
  kgen.func @simd_too_large(%arg0: !kgen.simd<65536, f32>) {
    hlcf.return
  }

  kgen.func @simd_pow2_ok(%arg0: !kgen.simd<4, f32>, %arg1: !kgen.scalar<index>) {
    hlcf.return
  }
}

// -----

// Layout math used to wrap oversized arrays to 0 / INT64_MIN bytes (issue 6848).
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64-i64:64:64", simd_bit_width=128, index_bit_width=64>} {
  // expected-error @below {{type is too large to fit in addressable memory, found '!pop.array<2305843009213693952, i64>'}}
  kgen.func @oversized_array(%arg0: !pop.array<2305843009213693952, i64>) {
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64-i64:64:64", simd_bit_width=128, index_bit_width=64>} {
  // expected-error @below {{type is too large to fit in addressable memory, found '!pop.array<1099511627776, array<1099511627776, i64>>'}}
  kgen.func @nested_oversized_array(%arg0: !pop.array<1099511627776, !pop.array<1099511627776, i64>>) {
    hlcf.return
  }
}

// -----

// 2^59 * 8 is still representable.
module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="e-p:64:64-i64:64:64", simd_bit_width=128, index_bit_width=64>} {
  kgen.func @large_but_representable_array(%arg0: !pop.array<576460752303423488, i64>) {
    hlcf.return
  }
}

