// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels \
// RUN:   --map-outline-structured-kernel-bodies --map-livm \
// RUN:   --map-flatten-shapes | FileCheck %s

// A rowwise kernel flattens like a loop: its dims merge where every capture
// lays them out one row after the other, read through collapse views. The
// reduced axis merges with neither neighbor, since each row is folded along
// it; the row and every fold's index drop to the merged rank.

// The dims before the axis merge into one row dim.
// CHECK-LABEL: mo.graph @inner_axis
// CHECK-DAG:   map.index.collapse.stride<#map.view.signature<[4, 6, 32] -> [24, 32]>, groups = [2, 1]> (%arg0)
// CHECK-DAG:   map.index.collapse.stride<#map.view.signature<[4, 6, 1] -> [24, 1]>, groups = [2, 1]> (%out)
// CHECK:       map.iter.rowwise.body<[24, 32], axis = 1>
// CHECK:       map.iter.reduce @{{.*}}ReduceSum index (%{{.*}}: !map.index<2>)
mo.graph @inner_axis(%x: !mo.tensor<[4, 6, 32], f32>) -> !mo.tensor<[4, 6, 1], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[4, 6, 32], f32>) -> !mo.tensor<[4, 6, 1], f32>
  mo.output %r : !mo.tensor<[4, 6, 1], f32>
}

// Dims on both sides of the axis merge, each side on its own; the axis moves
// to its group.
// CHECK-LABEL: mo.graph @middle_axis
// CHECK-DAG:   map.index.collapse.stride<#map.view.signature<[2, 3, 5, 4, 6] -> [6, 5, 24]>, groups = [2, 1, 2]> (%arg0)
// CHECK-DAG:   map.index.collapse.stride<#map.view.signature<[2, 3, 1, 4, 6] -> [6, 1, 24]>, groups = [2, 1, 2]> (%out)
// CHECK:       map.iter.rowwise.body<[6, 5, 24], axis = 1>
// CHECK:       map.iter.reduce @{{.*}}ReduceSum index (%{{.*}}: !map.index<3>)
mo.graph @middle_axis(%x: !mo.tensor<[2, 3, 5, 4, 6], f32>) -> !mo.tensor<[2, 3, 1, 4, 6], f32> {
  %r = mo.reduce.add(%x) : (!mo.tensor<[2, 3, 5, 4, 6], f32>) -> !mo.tensor<[2, 3, 1, 4, 6], f32>
  mo.output %r : !mo.tensor<[2, 3, 1, 4, 6], f32>
}

// A fused producer, a per-row side input and a symbolic product, declared as a
// graph parameter as for loops.
// CHECK-LABEL: mo.graph @sum_sq_bias
// CHECK:       kgen.param.declare param_flat_[[P:[0-9]+]]: scalar<si64> = <mul_no_wrap(N, 8)>
// CHECK-DAG:   map.index.collapse.stride<#map.view.signature<[N, 8, 64] -> [param_flat_[[P]], 64]>, groups = [2, 1]> (%arg0)
// CHECK-DAG:   map.index.collapse.stride<#map.view.signature<[N, 8, 1] -> [param_flat_[[P]], 1]>, groups = [2, 1]> (%arg1)
// CHECK:       map.iter.rowwise.body<[param_flat_[[P]], 64], axis = 1>
mo.graph @sum_sq_bias<N>(%x: !mo.tensor<[N, 8, 64], f32>, %b: !mo.tensor<[N, 8, 1], f32>) -> !mo.tensor<[N, 8, 1], f32> {
  %sq = mo.mul(%x, %x) : !mo.tensor<[N, 8, 64], f32>
  %r = mo.reduce.add(%sq) : (!mo.tensor<[N, 8, 64], f32>) -> !mo.tensor<[N, 8, 1], f32>
  %o = mo.add(%r, %b) : !mo.tensor<[N, 8, 1], f32>
  mo.output %o : !mo.tensor<[N, 8, 1], f32>
}

// A broadcast along dim 1 repeats its rows, so dims 0 and 1 don't merge and
// the kernel stays as it is.
// CHECK-LABEL: mo.graph @broadcast_blocks
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.rowwise.body<[4, 6, 32], axis = 2>
mo.graph @broadcast_blocks(%x: !mo.tensor<[4, 6, 32], f32>, %y: !mo.tensor<[4, 1, 32], f32>) -> !mo.tensor<[4, 6, 1], f32> {
  %yb = mo.static.broadcast_to(%y) : !mo.tensor<[4, 1, 32], f32> -> !mo.tensor<[4, 6, 32], f32>
  %s = mo.add(%x, %yb) : !mo.tensor<[4, 6, 32], f32>
  %r = mo.reduce.add(%s) : (!mo.tensor<[4, 6, 32], f32>) -> !mo.tensor<[4, 6, 1], f32>
  mo.output %r : !mo.tensor<[4, 6, 1], f32>
}
