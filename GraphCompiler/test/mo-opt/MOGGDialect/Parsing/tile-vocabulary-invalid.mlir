// RUN: mo-opt %s -split-input-file -verify-diagnostics

// One case per rule of the tile vocabulary's verifiers.

// expected-error @+1 {{expected a scalar dtype element type, got 'index'}}
func.func private @index_tile(%t: !mogg.tile<index>)

// -----

// expected-error @+1 {{expected a scalar dtype element type, got '!mogg.tile<f32>'}}
func.func private @nested_tile(%t: !mogg.tile<!mogg.tile<f32>>)

// -----

// expected-error @+1 {{expected a scalar dtype element type, got 'none'}}
func.func private @none_tile(%t: !mogg.tile<none>)

// -----

// expected-error @+1 {{expected '<'}}
func.func private @rankless_coord(%c: !mogg.tile_coord)

// -----

func.func @load_at_element_index(%x: !mogg.tensor, %idx: !map.index<2>) {
  // expected-error @+1 {{operand #1 must be}}
  %t = mogg.tile.load %x[%idx] : !mogg.tensor, !map.index<2> -> !mogg.tile<f32>
  return
}

// -----

func.func @store_at_element_index(%x: !mogg.tensor, %idx: !map.index<2>, %t: !mogg.tile<f32>) {
  // expected-error @+1 {{operand #1 must be}}
  mogg.tile.store %x[%idx] = %t : !mogg.tensor, !map.index<2>, !mogg.tile<f32>
  return
}

// -----

func.func @load_scalar(%x: !mogg.tensor, %coord: !mogg.tile_coord<2>) {
  // expected-error @+1 {{result #0 must be}}
  %v = mogg.tile.load %x[%coord] : !mogg.tensor, !mogg.tile_coord<2> -> f32
  return
}

// -----

func.func @store_scalar(%x: !mogg.tensor, %coord: !mogg.tile_coord<2>, %v: f32) {
  // expected-error @+1 {{operand #2 must be}}
  mogg.tile.store %x[%coord] = %v : !mogg.tensor, !mogg.tile_coord<2>, f32
  return
}

// -----

func.func @load_dtype_mismatch(%p: !mogg.pointer, %coord: !mogg.tile_coord<2>) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %x = mogg.tensor.create(%p, %s) <device = "gpu", elementType = f16, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{tile element type 'f32' does not match tensor element type 'f16'}}
  %t = mogg.tile.load %x[%coord] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
  return
}

// -----

func.func @store_dtype_mismatch(%p: !mogg.pointer, %coord: !mogg.tile_coord<2>, %t: !mogg.tile<f32>) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%p, %s) <device = "gpu", elementType = f16, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{tile element type 'f32' does not match tensor element type 'f16'}}
  mogg.tile.store %out[%coord] = %t : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  return
}

// -----

func.func @tile_call_loaded_scalar(%x: !mogg.tensor, %t: !mogg.tile<f32>, %idx: !map.index<2>) {
  %v = mogg.tensor.load %x[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
  // expected-error @+1 {{a tile call takes tiles or mogg.constant.scalar literals, got '!mogg.simd<f32>'}}
  %sum = mogg.call.elementwise @builtin_kernels.elementwise.Add(%t, %v) moOpName = "mo.add" : !mogg.tile<f32>, !mogg.simd<f32> -> !mogg.tile<f32>
  return
}

// -----

func.func @scalar_call_tile_input(%t: !mogg.tile<f32>, %v: f32) {
  // expected-error @+1 {{a tile input requires a tile result}}
  %sum = mogg.call.elementwise @builtin_kernels.elementwise.Add(%t, %v) moOpName = "mo.add" : !mogg.tile<f32>, f32 -> f32
  return
}

// -----

// The custom form parses a trailing operand as an input, so the index is set
// through the generic form.
func.func @tile_call_with_index(%t: !mogg.tile<f32>, %coord: !mogg.tile_coord<2>) {
  // expected-error @+1 {{a tile call takes no index}}
  %r = "mogg.call.elementwise"(%t, %coord) <{callee = @builtin_kernels.elementwise.Neg, moOpName = "mo.negative", operandSegmentSizes = array<i32: 1, 1>}> : (!mogg.tile<f32>, !mogg.tile_coord<2>) -> !mogg.tile<f32>
  return
}
