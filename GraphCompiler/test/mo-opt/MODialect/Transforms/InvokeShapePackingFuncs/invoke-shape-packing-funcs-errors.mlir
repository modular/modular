
// RUN: mo-opt %s -mo-invoke-shape-packing-funcs="use-host-system=false" --verify-diagnostics | FileCheck %s --enable-var-scope

#AVX2 = #M.target<
  triple = "x86_64-unknown-linux-gnu",
  arch = "alderlake",
  features = "+avx,+avx2,+avxvnni",
  simd_bit_width = 256,
  index_bit_width = 64
>

// CHECK-LABEL: mo.graph @matmul_packb_dynamic
// CHECK:       mo.invoke_shape_func["pack_matmul_b_shape_func"]
mo.graph @matmul_packb_dynamic(%arg0: !mo.tensor<[?, ?], f32>) attributes { target = #AVX2 } {

  // expected-remark@below {{unable to infer shapes for operation}}
  mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> MMPackSh1, MMPackSh2>(%arg0) [kgenParams = { a_shape = dense<-9223372036854775808> : vector<2xi64>, a_type = f32, b_shape = dense<-9223372036854775808> : vector<2xi64>, b_type = f32, c_shape = dense<-9223372036854775808> : vector<2xi64>, c_type = f32, transpose_in_0 = false }] : (!mo.tensor<[?, ?], f32>)
  mo.output
}

