// RUN: rm -rf %t && mkdir -p %t
// RUN: env MODULAR_DEBUG=ir-output-dir=%t mo-opt %s \
// RUN:   --mogg-import-packages --map-to-mogg --mogg-conform-call-operands \
// RUN:   --mogg-form-graph-blobs="max-calls=100" --mogg-lower-graph-blobs \
// RUN:   --canonicalize --mogg-outline-routines --mogg-emit-mojo \
// RUN:   -verify-each=false
// RUN: cat %t/*.mojo | FileCheck %s

// In a graph blob, a loop's width limit reaches `_graph_blob_foreach` as
// its `max_simd_width`, as it reaches `_foreach` in the loop's own launch. The
// kernel computes a run-time limit from the dims that cross the launch.

// CHECK:      def {{execute_device_[_0-9]+}}([[N:[A-Za-z0-9_]+]]: Int64,
// CHECK:      var [[LIMIT:[A-Za-z0-9_]+]] = layout.Coord(layout.Idx[8], Scalar[DType.int](Int([[N]])))
// CHECK:      var [[DIV:[A-Za-z0-9_]+]] = builtin_primitives.primitives.pow2_divisor_of_dims(std.utils.coord.coord_to_index_list([[LIMIT]]))
// CHECK-NEXT: builtin_primitives.primitives._graph_blob_foreach[dtypes = {{.+}}]({{[^,]+}}, {{[^,]+}}, max_simd_width = Scalar[DType.int]([[DIV]]))
// CHECK:      _graph_blob.barrier()
// CHECK:      builtin_primitives.primitives._graph_blob_foreach[dtypes = {{.+}}]({{[^,]+}}, {{[^,]+}}, max_simd_width = layout.Idx[2])
// CHECK:      mogg_launch_declarative
mo.graph @blob<N>(%a: !mo.tensor<[N], f32, gpu:0>) -> !mo.tensor<[N], f32, gpu:0> {
  %b = map.iter.opaque
      inputs (%a : !mo.tensor<[N], f32, gpu:0>)
      outputs (%out : !mo.tensor<[N], f32, gpu:0>) {
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[8, N]>)
        index (%idx: !map.index<1>)
        capturing {%a0 = %a : !mo.tensor<[N], f32, gpu:0>, %out0 = %out : !mo.tensor<[N], f32, gpu:0>} {
      %v = map.load %a0[%idx] : !mo.tensor<[N], f32, gpu:0>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32, gpu:0>
    }
  } {device = #M.device_ref<"gpu", 0>}
  %c = map.iter.opaque
      inputs (%b : !mo.tensor<[N], f32, gpu:0>)
      outputs (%out : !mo.tensor<[N], f32, gpu:0>) {
    map.iter.elementwise.body<[N]> max_simd_width(#map.gcd<[6, 10]>)
        index (%idx: !map.index<1>)
        capturing {%b0 = %b : !mo.tensor<[N], f32, gpu:0>, %out0 = %out : !mo.tensor<[N], f32, gpu:0>} {
      %v = map.load %b0[%idx] : !mo.tensor<[N], f32, gpu:0>
      map.store %out0[%idx] = %v : !mo.tensor<[N], f32, gpu:0>
    }
  } {device = #M.device_ref<"gpu", 0>}
  mo.output %c : !mo.tensor<[N], f32, gpu:0>
}
