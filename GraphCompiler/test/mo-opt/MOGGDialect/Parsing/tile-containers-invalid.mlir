// RUN: mo-opt %s -split-input-file -verify-diagnostics

// One case per rule of the tile container verifiers. The bodies are the
// interim scalar shape (`tensor.load` / `tensor.store`).

func.func @no_captures(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  // expected-error @+1 {{requires at least one (store-target) capture}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : !mogg.tile<f32>
  }
  return
}

// -----

func.func @arg_count(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{body takes a value, an index, and one arg per capture}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : !mogg.tile<f32>
  }
  return
}

// -----

func.func @compute_lane_without_yield(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a compute lane ends in a yield of one value}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.tile.store %cout[%idx] = %val : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  }
  return
}

// -----

func.func @store_lane_with_yield(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a store lane does not yield}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = store, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : !mogg.tile<f32>
  }
  return
}

// -----

func.func @store_lane_wrong_target(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %z = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a store lane ends in a store to the store-target capture at the loop index}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor, %cz = %z : !mogg.tensor} ptr %ptr lane = store, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.tile.store %cz[%idx] = %val : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  }
  return
}

// -----

func.func @load_off_index(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %z = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // The body is not isolated from above, so a tensor from the enclosing
  // scope can be named; it is not a capture.
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor, %cz = %z : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    // expected-error @+1 {{in a tile epilogue must read a capture at the loop index}}
    %zv = mogg.tile.load %z[%idx] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
    mogg.yield %zv : !mogg.tile<f32>
  }
  return
}

// -----

func.func @shape_captures(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a tile epilogue takes no shape captures}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor} shapes {%sh = %s} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : !mogg.tile<f32>
  }
  return
}

// -----

func.func @foreach_arg_count(%ptr: !mogg.pointer, %ctx: !mogg.context) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %x = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{body takes one arg per capture, then the index}}
  mogg.call.foreach.tile @builtin_primitives.primitives.foreach_fusion_tile shape(%s) context(%ctx) (%idx : !mogg.tile_coord<2>, %extra : !mogg.tile_coord<2>) capturing {%co = %out : !mogg.tensor, %cx = %x : !mogg.tensor} device = "gpu" {
    %xv = mogg.tile.load %cx[%idx] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
    mogg.tile.store %co[%idx] = %xv : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  }
  return
}

// -----

func.func @foreach_no_input(%ptr: !mogg.pointer, %ctx: !mogg.context) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{requires an output capture and at least one input capture}}
  mogg.call.foreach.tile @builtin_primitives.primitives.foreach_fusion_tile shape(%s) context(%ctx) (%idx : !mogg.tile_coord<2>) capturing {%co = %out : !mogg.tensor} device = "gpu" {
    %t = mogg.tile.load %co[%idx] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
    mogg.tile.store %co[%idx] = %t : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  }
  return
}

// -----

func.func @foreach_two_stores(%ptr: !mogg.pointer, %ctx: !mogg.context) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %x = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{body stores exactly once, got 2}}
  mogg.call.foreach.tile @builtin_primitives.primitives.foreach_fusion_tile shape(%s) context(%ctx) (%idx : !mogg.tile_coord<2>) capturing {%co = %out : !mogg.tensor, %cx = %x : !mogg.tensor} device = "gpu" {
    %xv = mogg.tile.load %cx[%idx] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
    mogg.tile.store %co[%idx] = %xv : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
    mogg.tile.store %co[%idx] = %xv : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  }
  return
}

// -----

func.func @foreach_store_off_index(%ptr: !mogg.pointer, %ctx: !mogg.context) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %x = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  mogg.call.foreach.tile @builtin_primitives.primitives.foreach_fusion_tile shape(%s) context(%ctx) (%idx : !mogg.tile_coord<2>) capturing {%co = %out : !mogg.tensor, %cx = %x : !mogg.tensor} device = "gpu" {
    %xv = mogg.tile.load %cx[%idx] : !mogg.tensor, !mogg.tile_coord<2> -> !mogg.tile<f32>
    // expected-error @+1 {{in a tile foreach must write a capture at the loop index}}
    mogg.tile.store %x[%idx] = %xv : !mogg.tensor, !mogg.tile_coord<2>, !mogg.tile<f32>
  }
  return
}

// -----

func.func @scalar_body(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %z = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor, %cz = %z : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    // expected-error @+1 {{in a tile epilogue must move a tile, with tile.load or tile.store}}
    %zv = mogg.tensor.load %cz[%idx] : !mogg.tensor, !mogg.tile_coord<2> -> f32
    mogg.yield %val : !mogg.tile<f32>
  }
  return
}

// -----

func.func @value_not_output_tile(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{the kernel's tile is a '!mogg.tile<f32>'}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<bf16>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.yield %val : !mogg.tile<bf16>
  }
  return
}

// -----

func.func @compute_lane_yields_other_tile(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // expected-error @+1 {{a compute lane yields a '!mogg.tile<f32>'}}
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.tile<f32>, %idx : !mogg.tile_coord<2>) capturing {%cout = %out : !mogg.tensor} ptr %ptr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    %c = mogg.constant.scalar 1.000000e+00 : f32 : f32
    mogg.yield %c : f32
  }
  return
}
