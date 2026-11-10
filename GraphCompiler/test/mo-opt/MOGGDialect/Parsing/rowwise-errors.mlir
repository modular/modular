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
    // expected-error @below{{body must yield one contribution}}
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

// -----

func.func @mapTwoArgs(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor} device = "cpu" {
    // expected-error @below{{body takes exactly one argument, the index}}
    mogg.rowwise.map (%i : !map.index<2>, %j : !map.index<2>) {
    }
  }
  return
}

// -----

func.func @mapOutsideRowwise() {
  // expected-error @below{{expects parent op 'mogg.call.rowwise'}}
  mogg.rowwise.map (%i : !map.index<2>) {
  }
  return
}

// -----

func.func @cacheLoadOutsidePhase(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor} device = "cpu" {
    // expected-error @below{{expects parent op to be one of 'mogg.rowwise.reduce, mogg.rowwise.map'}}
    %v = mogg.rowwise.cache.load : !mogg.simd<f32>
  }
  return
}

// -----

// Only the emit and the map phases store.
func.func @foldStores(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %copy: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %cc = %copy : !mogg.tensor} device = "cpu" {
    // expected-error @below{{stores; only the emit and the map phases store}}
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.tensor.store %cc[%i] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
  }
  return
}

// -----

// A store directly in the body writes the row.
func.func @storeOffRow(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor, %k: !map.index<2>) {
  // expected-error @below{{stores outside its row index, its emit and its map phases}}
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    mogg.tensor.store %co[%k] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// -----

// Every fold comes before the map phases.
func.func @foldAfterMap(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  // expected-error @below{{has a fold after a map phase}}
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    %a = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    mogg.rowwise.map (%j : !map.index<2>) {
      %v = mogg.tensor.load %cx[%j] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.tensor.store %co[%j] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
    %b = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
  }
  return
}

// -----

// A per-row store can't follow a map phase yet.
func.func @emitAndMap(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor, %sum: !mogg.tensor) {
  // expected-error @below{{has both per-row stores and a map phase}}
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cx = %x : !mogg.tensor, %co = %out : !mogg.tensor, %cs = %sum : !mogg.tensor} device = "cpu" {
    %acc = mogg.rowwise.reduce @builtin_kernels.monoids.ReduceSum (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cx[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
    mogg.rowwise.map (%j : !map.index<2>) {
      %v = mogg.tensor.load %cx[%j] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.tensor.store %co[%j] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
    mogg.rowwise.emit {
      mogg.tensor.store %cs[%row] = %acc : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
  }
  return
}

// -----

func.func @loadYieldType(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor} device = "cpu" {
    // expected-error @below{{body must yield one value of the result type}}
    %p = mogg.rowwise.cache (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cap[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f16>
      mogg.yield %v : !mogg.simd<f16>
    }
  }
  return
}

// -----

func.func @cacheNotFirst(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    mogg.rowwise.map (%j : !map.index<2>) {
      %v = mogg.tensor.load %cap[%j] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.tensor.store %co[%j] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
    }
    // expected-error @below{{must come first in the row, after constants only}}
    %p = mogg.rowwise.cache (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cap[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
  }
  return
}

// -----

func.func @cacheStores(%shape: !mogg.shape, %ctx: !mogg.context, %x: !mogg.tensor, %out: !mogg.tensor) {
  mogg.call.rowwise shape(%shape) axis(1) context(%ctx) (%row : !map.index<2>) capturing {%cap = %x : !mogg.tensor, %co = %out : !mogg.tensor} device = "cpu" {
    // expected-error @below{{must not store; it only reads the row}}
    %p = mogg.rowwise.cache (%i : !map.index<2>) -> !mogg.simd<f32> {
      %v = mogg.tensor.load %cap[%i] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
      mogg.tensor.store %co[%i] = %v : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
      mogg.yield %v : !mogg.simd<f32>
    }
  }
  return
}
