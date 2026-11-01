// RUN: mo-opt %s --map-fuse-horizontal | FileCheck %s

// An opaque kernel's result is returned and read, at its own index, by a
// foreach that already holds a `map.index.slice.select`. The foreach moves into
// the opaque's epilogue whole: the select stays, now at the epilogue's index,
// and its store keeps its place with the new destination. Written in MAP IR,
// since from MO the opaque path takes such a consumer into the epilogue before
// a select can form in it.

// CHECK-LABEL: mo.graph @opaque_feeds_select_kernel
// CHECK:       %[[O:[0-9]+]]:3 = map.iter.opaque
// CHECK:       map.iter.epilogue (%val : f32, %idx : !map.index<2>)
// CHECK:       map.index.slice.select<{{.*}}> (%idx) -> (%[[J:[a-z0-9]+]])
// CHECK-NEXT:  map.store %{{.*}}[%[[J]]] = %val
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[O]]#0, %[[O]]#1, %[[O]]#2
mogg.kernel.decl @kernels.relu registration = "kernels.relu" device_info = <"", "", "", ""> struct_params = [] is_user_defined = false
mo.graph @opaque_feeds_select_kernel(%X: !mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
  %r = map.iter.opaque inputs (%X : !mo.tensor<[8, 16], f32>) outputs (%out : !mo.tensor<[8, 16], f32>) {
    %epi = map.iter.epilogue (%val : f32, %idx : !map.index<2>) capturing {%cap = %out : !mo.tensor<[8, 16], f32>} {
      map.store %cap[%idx] = %val : !mo.tensor<[8, 16], f32>
    } : !mo.tensor<[8, 16], f32>
    map.call.opaque @kernels.relu[] inputs (%X) outputs (%epi) kernelName = "kernels.relu" : (!mo.tensor<[8, 16], f32>) -> (!mo.tensor<[8, 16], f32>)
    map.yield %out : !mo.tensor<[8, 16], f32>
  } {device = #M.device_ref<"cpu", 0>}
  %m:2 = map.iter.elementwise<[8, 16]> inputs (%r: !mo.tensor<[8, 16], f32>) outputs (%x: !mo.tensor<[8, 16], f32>, %y: !mo.tensor<[4, 16], f32>) index (%i: !map.index<2>) -> (!mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>) {
    %v = map.load %r[%i] : !mo.tensor<[8, 16], f32>
    map.store %x[%i] = %v : !mo.tensor<[8, 16], f32>
    map.index.slice.select<#map.index.slice.spec<[8, 16] -> [4, 16], starts = [1, 0], stops = [8, 16], steps = [2, 1]>> (%i) -> (%j) : !map.index<2> {
      map.store %y[%j] = %v : !mo.tensor<[4, 16], f32>
    }
  }
  mo.output %r, %m#0, %m#1 : !mo.tensor<[8, 16], f32>, !mo.tensor<[8, 16], f32>, !mo.tensor<[4, 16], f32>
}
