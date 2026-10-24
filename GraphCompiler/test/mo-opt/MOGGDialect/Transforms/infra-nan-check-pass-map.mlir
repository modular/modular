// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-elementwise --map-outline-structured-kernel-bodies --mo-nan-check-pass="stride=1" --map-to-mogg --cse | FileCheck %s --enable-var-scope

//===----------------------------------------------------------------------===//
// NanCheckPass runs at the MAP level: after fusion (so a check kernel is never
// fused into the kernel it checks) and before `--map-to-mogg`, which then
// compiles the injected `mo.custom` checks alongside the graph's own kernels.
// It walks `map.iter.elementwise` / `map.iter.opaque`, whose float results are the
// compiled-region boundaries; after `--map-to-mogg` every kernel here (the
// checked region and the two check kernels) is a `mogg.routine`.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: nan_check inserted after each compiled region (mogg.routine), not
// after each individual MO op. Three elementwise ops fuse into ONE kernel,
// so only ONE nan_check_count + nan_check_raise pair should appear.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @fused_elementwise
// CHECK:       mo.constant
// CHECK:       mo.constant
//     One fused kernel for all three elementwise ops:
// CHECK:       mogg.routine
//     nan_check_count reduction kernel:
// CHECK:       custom__nan_check_count
//     nan_check_raise kernel on host (no mo.transfer for CPU-only graphs):
// CHECK:       custom__nan_check_raise
//     Original tensor flows unchanged to output (no copy):
// CHECK:       mo.output

mo.graph @fused_elementwise() -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[4], f32>
  %1 = mo.constant <#M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[4], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[4], f32>
  %3 = mo.relu(%2) : !mo.tensor<[4], f32>
  %4 = mo.negative(%3) : !mo.tensor<[4], f32>
  mo.output %4 : !mo.tensor<[4], f32>
}

//===----------------------------------------------------------------------===//
// Test 2: Two separate compiled regions each get their own nan_check pair.
// A mo.custom op acts as a fusion boundary, producing two kernels.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @two_regions
// CHECK:       mo.constant
//     First kernel (add):
// CHECK:       mogg.routine
//     nan_check for first kernel:
// CHECK:       custom__nan_check_count
// CHECK:       custom__nan_check_raise
//     Second kernel (custom op boundary + relu):
// CHECK:       mogg.routine
//     nan_check for second kernel:
// CHECK:       custom__nan_check_count
// CHECK:       custom__nan_check_raise
// CHECK:       mo.output

mo.graph @two_regions(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mo.constant <#M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<4xf32>> {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[4], f32>
  %1 = mo.add(%arg0, %0) : !mo.tensor<[4], f32>
  %2 = mo.custom [symbol = "test_custom_op", device = #M.device_ref<"cpu", 0>](%1) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %3 = mo.relu(%2) : !mo.tensor<[4], f32>
  mo.output %3 : !mo.tensor<[4], f32>
}

//===----------------------------------------------------------------------===//
// Test 3: Integer tensor outputs are skipped — no nan_check for non-float.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @integer_skipped
// CHECK:       mogg.routine
// CHECK-NOT:   nan_check
// CHECK:       mo.output

mo.graph @integer_skipped(%arg0: !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32> {
  %0 = mo.relu(%arg0) : !mo.tensor<[4], si32>
  mo.output %0 : !mo.tensor<[4], si32>
}

//===----------------------------------------------------------------------===//
// Test 4: Mixed float and integer kernel results — only float gets nan_check.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @mixed_types
//     Float kernel result gets nan_check:
// CHECK:       mogg.routine
// CHECK:       custom__nan_check_count
//     Integer kernel result does NOT get its own nan_check_count:
// CHECK:       mogg.routine
// CHECK-NOT:   custom__nan_check_count
// CHECK:       mo.output

mo.graph @mixed_types(%arg0: !mo.tensor<[4], f32>, %arg1: !mo.tensor<[4], si32>) -> (!mo.tensor<[4], f32>, !mo.tensor<[4], si32>) {
  %0 = mo.relu(%arg0) : !mo.tensor<[4], f32>
  %1 = mo.relu(%arg1) : !mo.tensor<[4], si32>
  mo.output %0, %1 : !mo.tensor<[4], f32>, !mo.tensor<[4], si32>
}

//===----------------------------------------------------------------------===//
// Test 5: nan_check_raise kernel is present with label parameter.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @label_present
// CHECK:       mogg.routine
// CHECK:       custom__nan_check_count
// CHECK:       custom__nan_check_raise

mo.graph @label_present(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mo.relu(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}
