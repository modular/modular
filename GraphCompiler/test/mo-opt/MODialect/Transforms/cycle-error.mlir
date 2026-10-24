// RUN: mo-opt %s --mo-propagate-shapes --verify-diagnostics

// %4 reads D5, which is only bound by the rebind in %7, and %7 depends on %4
// through %5 and %6.
mo.graph @main<D1>(%arg0: !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si64>, !mo.tensor<[D5], bf16>) attributes {_kernel_library_paths = [], argument_names = ["input0"], result_names = ["output0", "output1"]} {
  %0 = mo.constant <#M.dense_array<1.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %1 = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  %2 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[1], si64>
  %3 = mo.add(%arg0, %2) : !mo.tensor<[1], si64>
  // expected-error@+1 {{cannot schedule operation: dependency cycle in the graph}}
  %4 = mosh.param.to_value: !mosh.ape = <[D5]>
  // expected-note@+1 {{also unschedulable}}
  %5 = mo.shape.to_tensor(%4) -> !mo.tensor<[D1], si64>
  // expected-note@+1 {{also unschedulable}}
  %6 = mo.random.normal<() -> RMO_D1>(%5, %1, %0, %arg0) : (!mo.tensor<[D1], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[RMO_D1], f32>
  // expected-note@+1 {{also unschedulable}}
  kgen.param.declare RMO_SH_EQ2: scalar<bool> = <eq(:scalar<si64> RMO_D1, 5)>
  // expected-note@+1 {{also unschedulable}}
  kgen.param.declare RMO_SH_CH3: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, RMO_SH_EQ2)>
  // expected-note@+1 {{also unschedulable}}
  %7 = mo.rebind<() -> D5>(%6) : !mo.tensor<[RMO_D1], f32> -> !mo.tensor<[D5], f32>
  // expected-note@+1 {{also unschedulable}}
  %8 = mo.cast(%7) : (!mo.tensor<[D5], f32>) -> !mo.tensor<[D5], bf16>
  // expected-note@+1 {{also unschedulable}}
  mo.output %3, %8 : !mo.tensor<[1], si64>, !mo.tensor<[D5], bf16>
} {counter = 3 : i64}
