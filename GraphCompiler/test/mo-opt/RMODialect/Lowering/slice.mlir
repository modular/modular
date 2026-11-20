// RUN: mo-opt --mo-legalize-rmo-operators="skip-rmo-rebinds" --mo-shape-inference %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @slice
mo.graph @slice<dim0>(%arg0: !mo.tensor<[dim0], f32>) -> !mo.tensor<[add(dim0, -1)], f32> attributes {argument_names = ["input0"], result_names = ["output0"]} {
  // CHECK: %0 = mo.constant <
  // CHECK: {{[[:space:]]}}mo.slice(%arg0, %0
  %1 = rmo.slice(%arg0) [starts = #mosh<ape[1]> : !mosh.ape, stops = #mosh<ape[dim0]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape] : (!mo.tensor<[dim0], f32>) -> !mo.tensor<[add(dim0, -1)], f32>
  mo.output %1 : !mo.tensor<[add(dim0, -1)], f32>
}
