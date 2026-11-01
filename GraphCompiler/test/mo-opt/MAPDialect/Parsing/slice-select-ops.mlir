// RUN: mo-opt %s | FileCheck %s
// RUN: mo-opt %s | mo-opt

// Round-trip map.index.slice.select beside the producer's own store, as
// horizontal fusion leaves it in an epilogue: once with static shapes and a
// strided outer dim, once with a symbolic source dim and a negative start, and
// once as a slice of a slice, two nested selects.

// map.call.opaque verifies its callee resolves to a mogg.kernel.decl.
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false

#stepSpec = #map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>
#windowSpec = #map.index.slice.spec<[N, 16] -> [4, 16], starts = [-4, 0], stops = [?, ?], steps = [1, 1]>
#innerSpec = #map.index.slice.spec<[4, 16] -> [2, 16], starts = [1, 0], stops = [4, 16], steps = [2, 1]>

// CHECK-LABEL: func.func @slice_select_step
// CHECK:       map.iter.epilogue (%[[VAL:[a-z0-9_]+]] : f32, %[[IDX:[a-z0-9_]+]] : !map.index<2>) capturing {%[[FULL:[a-z0-9_]+]] = %{{.*}} : !mo.tensor<[8, 16], f32>, %[[SLICE:[a-z0-9_]+]] = %{{.*}} : !mo.tensor<[4, 16], f32>} {
// CHECK-NEXT:    map.store %[[FULL]][%[[IDX]]] = %[[VAL]] : !mo.tensor<[8, 16], f32>
// CHECK-NEXT:    map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>> (%[[IDX]]) -> (%[[IOUT:[a-z0-9_]+]]) : !map.index<2> {
// CHECK-NEXT:      map.store %[[SLICE]][%[[IOUT]]] = %[[VAL]] : !mo.tensor<[4, 16], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  } : !mo.tensor<[8, 16], f32>
func.func @slice_select_step(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[4, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[4, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      map.index.slice.select<#stepSpec> (%idx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[4, 16], f32>
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}

// CHECK-LABEL: func.func @slice_select_window
// CHECK:       map.store %{{.*}} : !mo.tensor<[N, 16], f32>
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[N, 16] -> [4, 16], starts = [-4, 0], stops = [?, ?], steps = [1, 1]>> (%{{.*}}) -> (%[[IOUT:[a-z0-9_]+]]) : !map.index<2> {
// CHECK-NEXT:    map.store %{{.*}}[%[[IOUT]]] = %{{.*}} : !mo.tensor<[4, 16], f32>
func.func @slice_select_window(%X: !mo.tensor<[N, 16], f32>) -> (!mo.tensor<[N, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[N, 16], f32>) outputs (%full : !mo.tensor<[N, 16], f32>, %slice : !mo.tensor<[4, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[N, 16], f32>, %caps = %slice : !mo.tensor<[4, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[N, 16], f32>
      map.index.slice.select<#windowSpec> (%idx) -> (%iout) : !map.index<2> {
        map.store %caps[%iout] = %val : !mo.tensor<[4, 16], f32>
      }
    } : !mo.tensor<[N, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[N, 16], f32>) -> (!mo.tensor<[N, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[N, 16], f32>, !mo.tensor<[4, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[N, 16], f32>, !mo.tensor<[4, 16], f32>
}

// Rows 1, 3, 5, 7, then every other one of those from the second: the inner
// select's in-shape is the outer one's out-shape, and its body runs only where
// both select.
// CHECK-LABEL: func.func @slice_select_nested
// CHECK:       map.store %{{.*}} : !mo.tensor<[8, 16], f32>
// CHECK-NEXT:  map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>> (%{{.*}}) -> (%[[MID:[a-z0-9_]+]]) : !map.index<2> {
// CHECK-NEXT:    map.index.slice.select<#map.index.slice.spec<[4, 16] -> [2, 16], starts = [1, 0], stops = [4, 16], steps = [2, 1]>> (%[[MID]]) -> (%[[IOUT:[a-z0-9_]+]]) : !map.index<2> {
// CHECK-NEXT:      map.store %{{.*}}[%[[IOUT]]] = %{{.*}} : !mo.tensor<[2, 16], f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
func.func @slice_select_nested(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[2, 16], f32>) {
  %y:2 = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%full : !mo.tensor<[8, 16], f32>, %slice : !mo.tensor<[2, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%capf = %full : !mo.tensor<[8, 16], f32>, %caps = %slice : !mo.tensor<[2, 16], f32>} {
      map.store %capf[%idx] = %val : !mo.tensor<[8, 16], f32>
      map.index.slice.select<#stepSpec> (%idx) -> (%mid) : !map.index<2> {
        map.index.slice.select<#innerSpec> (%mid) -> (%iout) : !map.index<2> {
          map.store %caps[%iout] = %val : !mo.tensor<[2, 16], f32>
        }
      }
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %full, %slice : !mo.tensor<[8, 16], f32>, !mo.tensor<[2, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  return %y#0, %y#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[2, 16], f32>
}
