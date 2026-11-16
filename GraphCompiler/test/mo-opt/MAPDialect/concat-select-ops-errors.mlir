// RUN: mo-opt %s --split-input-file --verify-diagnostics

// map.index.concat.select's verifier: its spec's shapes, an untransformed
// index addressing the spec's out-shape, one region per input taking one index
// and yielding the op's results, and loads in a region at that input's shape.

// The axis extents must sum to the output's.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 25], f32> {
  %r = map.iter.elementwise<[4, 25]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[4, 25], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[4, 25], f32> {
    // expected-error @below{{input extents on axis 1 sum to 24, the output's is 25}}
    %v = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 25], axis = 1>> (%idx) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[4, 8], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      %x = map.load %b[%i] : !mo.tensor<[4, 16], f32>
      map.yield %x : f32
    }
    map.store %out[%idx] = %v : !mo.tensor<[4, 25], f32>
  }
  return %r : !mo.tensor<[4, 25], f32>
}

// -----

// The index must be the loop's own: a transformed one could carry a vector
// across inputs.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[24, 4], f32> {
  %r = map.iter.elementwise<[24, 4]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[24, 4], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[24, 4], f32> {
    %t = map.index.reshape<#map.index.reshape.spec<[4, 24] -> [24, 4]>> (%idx) : !map.index<2> -> !map.index<2>
    // expected-error @below{{index must be its foreach's loop index, a fold's index or an enclosing select's argument}}
    %v = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%t) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[4, 8], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      %x = map.load %b[%i] : !mo.tensor<[4, 16], f32>
      map.yield %x : f32
    }
    map.store %out[%idx] = %v : !mo.tensor<[24, 4], f32>
  }
  return %r : !mo.tensor<[24, 4], f32>
}

// -----

// The spec's out-shape must be the shape the loop index addresses.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[2, 24], f32> {
  %r = map.iter.elementwise<[2, 24]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[2, 24], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[2, 24], f32> {
    // expected-error @below{{does not match the shape}}
    %v = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%idx) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[4, 8], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      %x = map.load %b[%i] : !mo.tensor<[4, 16], f32>
      map.yield %x : f32
    }
    map.store %out[%idx] = %v : !mo.tensor<[2, 24], f32>
  }
  return %r : !mo.tensor<[2, 24], f32>
}

// -----

// One region per input.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %r = map.iter.elementwise<[4, 24]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[4, 24], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[4, 24], f32> {
    // expected-error @below{{expected one region per input, 2, got 1}}
    %v = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%idx) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[4, 8], f32>
      map.yield %x : f32
    }
    map.store %out[%idx] = %v : !mo.tensor<[4, 24], f32>
  }
  return %r : !mo.tensor<[4, 24], f32>
}

// -----

// A region yields one value of the op's result type.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %r = map.iter.elementwise<[4, 24]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[4, 24], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[4, 24], f32> {
    // expected-error @below{{region 1 must yield one value of the op's result type}}
    %v = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%idx) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[4, 8], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      map.yield
    }
    map.store %out[%idx] = %v : !mo.tensor<[4, 24], f32>
  }
  return %r : !mo.tensor<[4, 24], f32>
}

// -----

// A load at a region's argument reads that region's input.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %r = map.iter.elementwise<[4, 24]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[4, 24], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[4, 24], f32> {
    // expected-error @below{{loaded in region 0 does not match the input's shape}}
    %v = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%idx) : !map.index<2> -> f32 (%i: !map.index<2>) {
      %x = map.load %b[%i] : !mo.tensor<[4, 16], f32>
      map.yield %x : f32
    }, (%i: !map.index<2>) {
      %x = map.load %b[%i] : !mo.tensor<[4, 16], f32>
      map.yield %x : f32
    }
    map.store %out[%idx] = %v : !mo.tensor<[4, 24], f32>
  }
  return %r : !mo.tensor<[4, 24], f32>
}

// -----

// The op yields exactly one value: a result type is required.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %r = map.iter.elementwise<[4, 24]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[4, 24], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[4, 24], f32> {
    // expected-error @below{{expected '->'}}
    map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%idx) : !map.index<2> (%i: !map.index<2>) {
      map.yield
    }, (%i: !map.index<2>) {
      map.yield
    }
  }
  return %r : !mo.tensor<[4, 24], f32>
}

// -----

// The op yields exactly one value: a second result type is rejected.
func.func @f(%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[4, 24], f32> {
  %r = map.iter.elementwise<[4, 24]>
  inputs  (%a: !mo.tensor<[4, 8], f32>, %b: !mo.tensor<[4, 16], f32>)
  outputs (%out: !mo.tensor<[4, 24], f32>)
  index   (%idx: !map.index<2>) -> !mo.tensor<[4, 24], f32> {
    // expected-error @below{{expected '(' in argument list}}
    %v:2 = map.index.concat.select<#map.index.concat.spec<[4, 8], [4, 16] -> [4, 24], axis = 1>> (%idx) : !map.index<2> -> f32, f32 (%i: !map.index<2>) {
      %x = map.load %a[%i] : !mo.tensor<[4, 8], f32>
      map.yield %x, %x : f32, f32
    }, (%i: !map.index<2>) {
      %y = map.load %b[%i] : !mo.tensor<[4, 16], f32>
      map.yield %y, %y : f32, f32
    }
    map.store %out[%idx] = %v#0 : !mo.tensor<[4, 24], f32>
  }
  return %r : !mo.tensor<[4, 24], f32>
}
