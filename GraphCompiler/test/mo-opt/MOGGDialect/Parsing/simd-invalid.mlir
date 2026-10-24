// RUN: mo-opt %s -split-input-file -verify-diagnostics

// One case per rule that keeps builtin scalars out of SIMD fusion bodies.

// expected-error @+1 {{expected a scalar dtype element type, got 'index'}}
func.func private @index_simd(%v: !mogg.simd<index>)

// -----

// expected-error @+1 {{expected a scalar dtype element type, got '!mogg.simd<f32>'}}
func.func private @nested_simd(%v: !mogg.simd<!mogg.simd<f32>>)

// -----

func.func @scalar_simd_call(%x: !mogg.tensor, %idx: !map.index<2>, %s: f32) {
  %v = mogg.tensor.load %x[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
  // expected-error @+1 {{a SIMD call takes and returns !mogg.simd values, got 'f32'}}
  %sum = mogg.call.elementwise @builtin_kernels.elementwise.Add(%v, %s) moOpName = "mo.add" : !mogg.simd<f32>, f32 -> !mogg.simd<f32>
  return
}

// -----

func.func @scalar_literal_in_simd_call(%x: !mogg.tensor, %idx: !map.index<2>) {
  %v = mogg.tensor.load %x[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
  // expected-error @+1 {{a scalar result may only feed a tile call; use !mogg.simd in a SIMD body}}
  %c = mogg.constant.scalar 2.000000e+00 : f32 : f32
  %sum = mogg.call.elementwise @builtin_kernels.elementwise.Add(%v, %c) moOpName = "mo.add" : !mogg.simd<f32>, f32 -> !mogg.simd<f32>
  return
}

// -----

func.func @literal_type_mismatch() {
  // expected-error @+1 {{value attribute type 'f16' does not match element type 'f32'}}
  %c = mogg.constant.scalar 2.000000e+00 : f16 : !mogg.simd<f32>
  return
}

// -----

func.func @scalar_epilogue_value(%ptr: !mogg.pointer, %cp: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
  %t = mogg.tensor.create(%ptr, %s) <device = "cpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{the value argument must be a !mogg.simd, got 'f32'}}
  %epi = mogg.tensor.create.epilogue %s (%val : f32, %i : !map.index<2>) capturing {%cap = %t : !mogg.tensor} ptr %cp
      device = "cpu", elementType = f32, io = Output, mut = true {
    %v = mogg.tensor.load %cap[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
    mogg.tensor.store %cap[%i] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// -----

func.func @scalar_rowwise_contribution(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %f: f32) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor} device = "cpu" {
    // expected-error @below{{body must yield !mogg.simd contributions, got 'f32'}}
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      mogg.yield %f : f32
    }
  }
  return
}

// -----

func.func @scalar_prologue_value(%ptr: !mogg.pointer, %f: f32) {
  %s = mogg.shape.create() <shape = #mosh<ape[2, 2]> : !mosh.ape>
  %t = mogg.tensor.create(%ptr, %s) <device = "cpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[2, 2]> : !mosh.ape> : !mogg.tensor
  // expected-error @below{{body must yield the !mogg.simd value the kernel reads, got 'f32'}}
  %pro = mogg.tensor.create.prologue %s (%i : !map.index<2>) capturing {%cap = %t : !mogg.tensor} device = "cpu", elementType = f32, io = Input, mut = false {
    mogg.yield %f : f32
  }
  return
}

// -----

func.func @simd_literal_in_tile_call(%t: !mogg.tile<f32>) {
  %c = mogg.constant.scalar 2.000000e+00 : f32 : !mogg.simd<f32>
  // expected-error @below{{a tile call takes tiles or mogg.constant.scalar literals, got '!mogg.simd<f32>'}}
  %sum = mogg.call.elementwise @builtin_kernels.elementwise.Add(%t, %c) moOpName = "mo.add" : !mogg.tile<f32>, !mogg.simd<f32> -> !mogg.tile<f32>
  return
}
