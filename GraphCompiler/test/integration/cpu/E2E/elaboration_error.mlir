// RUN: mkdir -p %t_mojo
// RUN: mojo precompile %S/inputs/TestPck -o %t_mojo/testpck.mojoc
// RUN: not mo-opt --gc-pipeline=kernel-library-paths="%t_mojo/testpck.mojoc" %s -verify-diagnostics 2>&1 | FileCheck %s

// CHECK: error: function instantiation failed
// CHECK: error: The graph compiler could not elaborate the generated KGEN

mo.graph @elaboration_error_in_kernel() -> !mo.tensor<[1], si32> {
  %0 = mo.custom [symbol = "fails_to_elaborate", device = #M.device_ref<"cpu", 0>]() : () -> (!mo.tensor<[1], si32>)
  mo.output %0 : !mo.tensor<[1], si32>
}
