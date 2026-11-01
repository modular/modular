// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip the tile fusion container ops. The bodies are the interim scalar
// shape (`tensor.load` / `tensor.store`); they switch to the tile vocabulary
// once it lands.

// CHECK-LABEL: func.func @compute_epilogue
func.func @compute_epilogue(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %z = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %optr = mogg.tensor.get_ptr(%out) : !mogg.tensor
  // CHECK: mogg.tile.create.epilogue %{{.*}} (%[[VAL:.*]] : !mogg.simd<f32>, %[[IDX:.*]] : !map.index<2>) capturing {%[[COUT:.*]] = %{{.*}} : !mogg.tensor, %[[CZ:.*]] = %{{.*}} : !mogg.tensor} ptr %{{.*}} lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
  // CHECK-NEXT:   %[[ZV:.*]] = mogg.tensor.load %[[CZ]][%[[IDX]]]
  // CHECK-NEXT:   mogg.yield %[[ZV]] : !mogg.simd<f32>
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.simd<f32>, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor, %cz = %z : !mogg.tensor} ptr %optr lane = compute, device = "gpu", elementType = f32, io = Output, mut = true {
    %zv = mogg.tensor.load %cz[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
    mogg.yield %zv : !mogg.simd<f32>
  }
  return
}

// A store lane owns the store into capture 0.
// CHECK-LABEL: func.func @store_epilogue
func.func @store_epilogue(%ptr: !mogg.pointer) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %optr = mogg.tensor.get_ptr(%out) : !mogg.tensor
  // CHECK: mogg.tile.create.epilogue %{{.*}} (%[[VAL:.*]] : !mogg.simd<f32>, %[[IDX:.*]] : !map.index<2>) capturing {%[[COUT:.*]] = %{{.*}} : !mogg.tensor} ptr %{{.*}} lane = store, device = "gpu", elementType = f32, io = Output, mut = true {
  // CHECK-NEXT:   mogg.tensor.store %[[COUT]][%[[IDX]]] = %[[VAL]]
  %epi = mogg.tile.create.epilogue %s (%val : !mogg.simd<f32>, %idx : !map.index<2>) capturing {%cout = %out : !mogg.tensor} ptr %optr lane = store, device = "gpu", elementType = f32, io = Output, mut = true {
    mogg.tensor.store %cout[%idx] = %val : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// CHECK-LABEL: func.func @foreach_tile
func.func @foreach_tile(%ptr: !mogg.pointer, %ctx: !mogg.context) {
  %s = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %out = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %x = mogg.tensor.create(%ptr, %s) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // CHECK: mogg.call.foreach.tile @builtin_primitives.primitives.foreach_fusion_tile shape(%{{.*}}) context(%{{.*}}) (%[[IDX:.*]] : !map.index<2>) capturing {%[[CO:.*]] = %{{.*}} : !mogg.tensor, %[[CX:.*]] = %{{.*}} : !mogg.tensor} device = "gpu" {
  // CHECK-NEXT:   %[[XV:.*]] = mogg.tensor.load %[[CX]][%[[IDX]]]
  // CHECK-NEXT:   mogg.tensor.store %[[CO]][%[[IDX]]] = %[[XV]]
  mogg.call.foreach.tile @builtin_primitives.primitives.foreach_fusion_tile shape(%s) context(%ctx) (%idx : !map.index<2>) capturing {%co = %out : !mogg.tensor, %cx = %x : !mogg.tensor} device = "gpu" {
    %xv = mogg.tensor.load %cx[%idx] : !mogg.tensor, !map.index<2> -> !mogg.simd<f32>
    mogg.tensor.store %co[%idx] = %xv : !mogg.tensor, !map.index<2>, !mogg.simd<f32>
  }
  return
}

// CHECK-LABEL: func.func @view_copy
func.func @view_copy(%ptr: !mogg.pointer, %ctx: !mogg.context) {
  %s_in = mogg.shape.create() <shape = #mosh<ape[8]> : !mosh.ape>
  %s_out = mogg.shape.create() <shape = #mosh<ape[16, 8]> : !mosh.ape>
  %in = mogg.tensor.create(%ptr, %s_in) <device = "gpu", elementType = f32, io = Input, mut = false, shapeAttr = #mosh<ape[8]> : !mosh.ape> : !mogg.tensor
  %out = mogg.tensor.create(%ptr, %s_out) <device = "gpu", elementType = f32, io = Output, mut = true, shapeAttr = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  %bcast = mogg.tensor.create.broadcast(%in) <elementType = f32, mut = false, outShape = #mosh<ape[16, 8]> : !mosh.ape> : !mogg.tensor
  // CHECK: mogg.call.view_copy @builtin_primitives.primitives.view_copy_impl(%{{.*}}, %{{.*}} : !mogg.tensor, !mogg.tensor) context(%{{.*}}) device = "gpu"
  mogg.call.view_copy @builtin_primitives.primitives.view_copy_impl(%out, %bcast : !mogg.tensor, !mogg.tensor) context(%ctx) device = "gpu"
  return
}
