// RUN: mo-opt %s -split-input-file -verify-diagnostics

func.func @noResult(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor} device = "cpu" {
    // expected-error @below{{must have at least one result}}
    mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) {
      %v = mogg.tensor.load %cap[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
  }
  return
}

// -----

func.func @noContribution(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor} device = "cpu" {
    // expected-error @below{{body must yield at least one contribution}}
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      mogg.yield
    }
  }
  return
}

// -----

func.func @outsideRowwise() {
  // expected-error @below{{expects parent op 'mogg.call.rowwise'}}
  %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
    %c = mogg.constant.scalar 1.0 : f32 : !mogg.simd<f32>
    mogg.yield %c : !mogg.simd<f32>
  }
  return
}
