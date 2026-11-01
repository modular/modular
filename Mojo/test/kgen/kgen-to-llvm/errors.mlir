// RUN: kgen-opt -lower-kgen-to-llvm -verify-diagnostics -split-input-file %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{failed to convert func signature}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func @unsupported(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    hlcf.return %arg0 : tensor<4xf32>
  }
}

// -----

// expected-error @below {{could not find an enclosing target specification}}
module {
  kgen.func @no_target() {
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=64>} {
  // expected-error @below {{'"unknown_attribute"' is not defined}}
  // expected-error @below {{failed to legalize operation 'kgen.func' that was explicitly marked illegal}}
  kgen.func export @llvm_metadata() attributes {
    fnAttrs = {
      unknown_attribute = #pop.array<256, 1, 4> : !pop.array<3, i32>
    }
  } {
    hlcf.return
  }
}
