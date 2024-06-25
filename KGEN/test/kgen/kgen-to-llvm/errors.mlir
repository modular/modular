// RUN: kgen-opt -lower-kgen-to-llvm -verify-diagnostics -split-input-file %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error@+2 {{failed to convert func signature}}
  // expected-error@+1 {{failed to legalize operation 'kgen.func'}}
  kgen.func @unsupported(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    kgen.return %arg0 : tensor<4xf32>
  }
}

// -----

// expected-error @below {{could not find an enclosing target specification}}
module {
  kgen.func @no_target() {
    kgen.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="skylake-avx512", features="+fma", data_layout="", simd_bit_width=128, tune_cpu="skylake-avx512">} {
kgen.func @rebind(%arg0: index) -> f32 {
  // expected-error @below {{invalid rebind between two unequal types: index to f32}}
  // expected-error @below {{failed to legalize operation}}
  %0 = kgen.rebind %arg0 : index to f32
  kgen.return %0 : f32
}
}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  kgen.func @capturing_parameter_closure(%arg0: !kgen.pointer<string>) capturing -> !kgen.string {
    %0 = pop.load %arg0 : !kgen.pointer<string>
    kgen.return %0 : !kgen.string
  }

  kgen.func @materializing_user() -> !kgen.signature<(!kgen.pointer<string>) capturing -> !kgen.string> {
    // expected-error @below {{capturing closures cannot be materialized as runtime values}}
    // expected-error @below {{failed to legalize operation}}
    %struct = kgen.param.constant: struct<((!kgen.pointer<string>) capturing -> !kgen.string, index)> = <{@capturing_parameter_closure, 8}>
    %0 = kgen.struct.extract %struct[0] : !kgen.struct<((!kgen.pointer<string>) capturing -> !kgen.string, index)>
    kgen.return %0 : !kgen.signature<(!kgen.pointer<string>) capturing -> !kgen.string>
  }
}
