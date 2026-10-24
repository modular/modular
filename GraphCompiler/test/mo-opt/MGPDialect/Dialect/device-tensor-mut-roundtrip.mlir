// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s

// Round-trip of the optional `mut` and `stable` markers on !mgp.tensor. The
// markers are lowered forms of !mo.buffer and stable-address semantics;
// unmarked tensors must print exactly as before.

// CHECK-LABEL: func.func @mut_type_round_trip
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<4x4xf32, mut>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<?x?xf32, mut>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<8xf32>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<8xf32, <"gpu", 1>>
func.func @mut_type_round_trip(
    %host_mut: !mgp.tensor<4x4xf32, mut>,
    %gpu_mut: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut>,
    %dyn_mut: !mgp.tensor<?x?xf32, mut>,
    %plain: !mgp.tensor<8xf32>,
    %gpu_plain: !mgp.tensor<8xf32, <"gpu", 1>>) {
  mef.output
}

// CHECK-LABEL: func.func @stable_type_round_trip
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<4x4xf32, stable>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<464x4800xf16, <"gpu", 0>, stable>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<?x?xf32, stable>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<8xf32>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<8xf32, <"gpu", 1>>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<4x4xf32, mut, stable>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut, stable>
func.func @stable_type_round_trip(
    %host_stable: !mgp.tensor<4x4xf32, stable>,
    %gpu_stable: !mgp.tensor<464x4800xf16, <"gpu", 0>, stable>,
    %dyn_stable: !mgp.tensor<?x?xf32, stable>,
    %plain: !mgp.tensor<8xf32>,
    %gpu_plain: !mgp.tensor<8xf32, <"gpu", 1>>,
    %host_mut_stable: !mgp.tensor<4x4xf32, mut, stable>,
    %gpu_mut_stable: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut, stable>) {
  mef.output
}

// The parser accepts the device, mut, and stable flags in any order, but the
// printer always emits them in canonical device/mut/stable order.
// CHECK-LABEL: func.func @stable_type_round_trip_any_order
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<4x4xf32, mut, stable>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<4x4xf32, mut, stable>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut, stable>
// CHECK-SAME: %{{[a-zA-Z0-9_]+}}: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut, stable>
func.func @stable_type_round_trip_any_order(
    %host_mut_then_stable: !mgp.tensor<4x4xf32, mut, stable>,
    %host_stable_then_mut: !mgp.tensor<4x4xf32, stable, mut>,
    %gpu_mut_then_stable: !mgp.tensor<464x4800xf16, <"gpu", 0>, mut, stable>,
    %gpu_stable_then_mut: !mgp.tensor<464x4800xf16, <"gpu", 0>, stable, mut>) {
  mef.output
}

// CHECK-LABEL: func.func @tensor_create_mut
func.func @tensor_create_mut(
    %ch: !mef.chain,
    %buf: !mgp.buffer<<"gpu", 0>>,
    %spec: !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"gpu", 0>, mut> {
  // CHECK: mgp.tensor.create %{{.*}}, %{{.*}}, %{{.*}} mut : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
  %0 = mgp.tensor.create %ch, %buf, %spec mut : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
  mef.output %0 : !mgp.tensor<3x5xf32, <"gpu", 0>, mut>
}

// The slice result derives its mutability from the sliced tensor.
// CHECK-LABEL: func.func @tensor_slice_propagates_mut
func.func @tensor_slice_propagates_mut(
    %input: !mgp.tensor<5xf32, <"gpu", 0>, mut>,
    %spec: !mgp.tensor_spec<3xf32>,
    %start: !mgp.tensor<1xsi64>,
    %dctx: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<3xf32, <"gpu", 0>, mut> {
  // CHECK: mgp.tensor.slice
  %sliced = mgp.tensor.slice %input, %spec, %start, %dctx : !mgp.tensor<5xf32, <"gpu", 0>, mut>, !mgp.tensor_spec<3xf32>, !mgp.tensor<1xsi64>, !mgp.device_context<<"gpu", 0>>
  // CHECK: mef.output %{{.*}} : !mgp.tensor<3xf32, <"gpu", 0>, mut>
  mef.output %sliced : !mgp.tensor<3xf32, <"gpu", 0>, mut>
}

// CHECK-LABEL: func.func @tensor_create_stable
func.func @tensor_create_stable(
    %ch: !mef.chain,
    %buf: !mgp.buffer<<"gpu", 0>>,
    %spec: !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"gpu", 0>, stable> {
  // CHECK: mgp.tensor.create %{{.*}}, %{{.*}}, %{{.*}} stable : !mgp.tensor<3x5xf32, <"gpu", 0>, stable>
  %0 = mgp.tensor.create %ch, %buf, %spec stable : !mgp.tensor<3x5xf32, <"gpu", 0>, stable>
  mef.output %0 : !mgp.tensor<3x5xf32, <"gpu", 0>, stable>
}

// The slice result also derives its stability from the sliced tensor.
// CHECK-LABEL: func.func @tensor_slice_propagates_stable
func.func @tensor_slice_propagates_stable(
    %input: !mgp.tensor<5xf32, <"gpu", 0>, stable>,
    %spec: !mgp.tensor_spec<3xf32>,
    %start: !mgp.tensor<1xsi64>,
    %dctx: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<3xf32, <"gpu", 0>, stable> {
  // CHECK: mgp.tensor.slice
  %sliced = mgp.tensor.slice %input, %spec, %start, %dctx : !mgp.tensor<5xf32, <"gpu", 0>, stable>, !mgp.tensor_spec<3xf32>, !mgp.tensor<1xsi64>, !mgp.device_context<<"gpu", 0>>
  // CHECK: mef.output %{{.*}} : !mgp.tensor<3xf32, <"gpu", 0>, stable>
  mef.output %sliced : !mgp.tensor<3xf32, <"gpu", 0>, stable>
}

// CHECK-LABEL: func.func @tensor_create_mut_stable
func.func @tensor_create_mut_stable(
    %ch: !mef.chain,
    %buf: !mgp.buffer<<"gpu", 0>>,
    %spec: !mgp.tensor_spec<3x5xf32>) -> !mgp.tensor<3x5xf32, <"gpu", 0>, mut, stable> {
  // CHECK: mgp.tensor.create %{{.*}}, %{{.*}}, %{{.*}} mut stable : !mgp.tensor<3x5xf32, <"gpu", 0>, mut, stable>
  %0 = mgp.tensor.create %ch, %buf, %spec mut stable : !mgp.tensor<3x5xf32, <"gpu", 0>, mut, stable>
  mef.output %0 : !mgp.tensor<3x5xf32, <"gpu", 0>, mut, stable>
}
