// RUN: kgen-opt -split-input-file -emit-llvm %s | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: define internal float @trivial
kgen.func @trivial(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  kgen.return %arg0 : !pop.simd<1, f32>
}

// CHECK-LABEL: define internal void @no_result
kgen.func @no_result(%arg0: !pop.simd<1, f32>) {
  kgen.return
}

// CHECK-LABEL: define internal { float, float } @two_results
kgen.func @two_results(%arg0: !pop.simd<1, f32>) -> (!pop.simd<1, f32>, !pop.simd<1, f32>) {
  kgen.return %arg0, %arg0 : !pop.simd<1, f32>, !pop.simd<1, f32>
}

kgen.func @empty_struct_result() -> !pop.struct<> {
  %0 = pop.struct.create() : !pop.struct<>
  kgen.return %0 : !pop.struct<>
}

kgen.func @call_struct_result() -> !pop.struct<> {
  %0 = kgen.call @empty_struct_result() : () -> !pop.struct<>
  kgen.return %0 : !pop.struct<>
}


// CHECK-LABEL: define void @convert_call
kgen.func @convert_call(%arg0: !pop.simd<1, f32>) {
  %0 = kgen.call @trivial(%arg0) : (!pop.simd<1, f32>) -> !pop.simd<1, f32>
  kgen.call @no_result(%arg0) : (!pop.simd<1, f32>) -> ()
  %1:2 = kgen.call @two_results(%arg0) : (!pop.simd<1, f32>) -> (!pop.simd<1, f32>, !pop.simd<1, f32>)
  kgen.return
}

kgen.export @convert_call
kgen.export @call_struct_result

}
