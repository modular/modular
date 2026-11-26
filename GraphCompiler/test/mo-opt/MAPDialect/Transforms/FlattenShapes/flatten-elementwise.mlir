// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels \
// RUN:   --map-outline-structured-kernel-bodies --map-livm \
// RUN:   --map-flatten-shapes | FileCheck %s
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-to-map --map-fuse-kernels \
// RUN:   --map-outline-structured-kernel-bodies --map-livm --canonicalize \
// RUN:   --map-flatten-shapes | FileCheck %s --check-prefix=FOLDED

// `map-flatten-shapes` merges adjacent loop dims that every capture lays out
// one row after the other, reading each capture through a collapse view.

// Contiguous captures: the loop runs over one dim.
// CHECK-LABEL: mo.graph @contiguous
// CHECK:       map.index.collapse.stride<#map.view.signature<[4, 6] -> [24]>, groups = [2]>
// CHECK:       map.iter.elementwise.body<[24]>
// CHECK-NEXT:  index (%{{.*}}: !map.index<1>)
mo.graph @contiguous(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 6], f32>) -> !mo.tensor<[4, 6], f32> {
  %x = mo.add(%a, %b) : !mo.tensor<[4, 6], f32>
  %y = mo.relu(%x) : !mo.tensor<[4, 6], f32>
  mo.output %y : !mo.tensor<[4, 6], f32>
}

// A symbolic merged extent becomes a graph parameter, so the kernel can name
// it.
// CHECK-LABEL: mo.graph @dynamic
// CHECK:       kgen.param.declare [[P:param_flat_[0-9]+]]: scalar<si64> = <mul_no_wrap(T, 4)>
// CHECK:       map.iter.elementwise.body<[[[P]]]>
mo.graph @dynamic<T>(%a: !mo.tensor<[T, 4], f32>) -> !mo.tensor<[T, 4], f32> {
  %y = mo.relu(%a) : !mo.tensor<[T, 4], f32>
  mo.output %y : !mo.tensor<[T, 4], f32>
}

// A constant folded into the kernel (`map.constant`, from canonicalization)
// is a row-major tensor read like any other capture, so the loop still merges
// its dims.
// FOLDED-LABEL: mo.graph @constant_capture
// FOLDED:       map.iter.opaque
// FOLDED:       [[C:%.+]] = map.constant
// FOLDED:       map.index.collapse.stride<#map.view.signature<[2, 4] -> [8]>, groups = [2]> ([[C]])
// FOLDED:       map.iter.elementwise.body<[8]>
mo.graph @constant_capture(%a: !mo.tensor<[2, 4], f32>) -> !mo.tensor<[2, 4], f32> {
  %c = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<2x4xf32>> : !mo.tensor<[2, 4], f32>
  %x = mo.add(%a, %c) : !mo.tensor<[2, 4], f32>
  mo.output %x : !mo.tensor<[2, 4], f32>
}

// A slice whose stop is only known at run time may cut a dim with an unknown
// extent (`?`), even when its in and out extents print the same, so that dim
// doesn't merge with the dim before it. Dims with a static extent are whole,
// so dims 1 and 2 still merge.
// CHECK-LABEL: mo.graph @runtime_stop
// CHECK:       map.iter.elementwise.body<[4, param_flat_{{[0-9]+}}]>
mo.graph @runtime_stop(%x: !mo.tensor<[4, ?, 16], f32>, %stop: !mo.tensor<[3], si64>) -> !mo.tensor<[4, ?, 16], f32> {
  %start = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %step = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %s = mo.slice(%x, %start, %stop, %step) : (!mo.tensor<[4, ?, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, ?, 16], f32>
  %y = mo.relu(%s) : !mo.tensor<[4, ?, 16], f32>
  mo.output %y : !mo.tensor<[4, ?, 16], f32>
}

// A bias broadcast over the two outer dims replicates both (stride 0), so
// they merge with each other but not with the innermost.
// CHECK-LABEL: mo.graph @bias
// CHECK:       map.iter.elementwise.body<[32, 16]>
mo.graph @bias(%a: !mo.tensor<[4, 8, 16], f32>, %b: !mo.tensor<[16], f32>) -> !mo.tensor<[4, 8, 16], f32> {
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[16], f32> -> !mo.tensor<[4, 8, 16], f32>
  %y = mo.add(%a, %bb) : !mo.tensor<[4, 8, 16], f32>
  mo.output %y : !mo.tensor<[4, 8, 16], f32>
}

// A transpose swapping the two dims breaks the pair.
// CHECK-LABEL: mo.graph @transpose
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.elementwise.body<[16, 4]>
mo.graph @transpose(%a: !mo.tensor<[4, 16], f32>) -> !mo.tensor<[16, 4], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%a, %perm) : (!mo.tensor<[4, 16], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[16, 4], f32>
  %y = mo.relu(%t) : !mo.tensor<[16, 4], f32>
  mo.output %y : !mo.tensor<[16, 4], f32>
}

// A window cut on the middle dim: the middle and innermost dims still merge
// (the innermost is whole), the outer one doesn't.
// CHECK-LABEL: mo.graph @window
// CHECK:       map.iter.elementwise.body<[4, 64]>
mo.graph @window(%a: !mo.tensor<[4, 8, 16], f32>) -> !mo.tensor<[4, 4, 16], f32> {
  %starts = mo.constant <#M.dense_array<0, 2, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 6, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %w = mo.slice(%a, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %y : !mo.tensor<[4, 4, 16], f32>
}

// A concat along the innermost axis: the dim before the axis interleaves the
// inputs, so nothing merges.
// CHECK-LABEL: mo.graph @concat
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.elementwise.body<[4, 16]>
mo.graph @concat(%a: !mo.tensor<[4, 6], f32>, %b: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32> {
  %r = mo.concat[1] (%a, %b) : (!mo.tensor<[4, 6], f32>, !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 16], f32>
  mo.output %r : !mo.tensor<[4, 16], f32>
}

// A concat along the outer axis merges with the dims after it: the select
// reads merged extents 48 and 80, and each region reads its input through a
// collapse view.
// CHECK-LABEL: mo.graph @concat_outer
// CHECK:       map.index.collapse.stride<#map.view.signature<[3, 16] -> [48]>, groups = [2]>
// CHECK:       map.index.collapse.stride<#map.view.signature<[5, 16] -> [80]>, groups = [2]>
// CHECK:       map.iter.elementwise.body<[128]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[48], [80] -> [128], axis = 0>>
mo.graph @concat_outer(%a: !mo.tensor<[3, 16], f32>, %b: !mo.tensor<[5, 16], f32>) -> !mo.tensor<[8, 16], f32> {
  %r = mo.concat[0] (%a, %b) : (!mo.tensor<[3, 16], f32>, !mo.tensor<[5, 16], f32>) -> !mo.tensor<[8, 16], f32>
  mo.output %r : !mo.tensor<[8, 16], f32>
}

// The same with rows of 5.
// CHECK-LABEL: mo.graph @concat_outer_odd
// CHECK:       map.iter.elementwise.body<[40]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[15], [25] -> [40], axis = 0>>
mo.graph @concat_outer_odd(%a: !mo.tensor<[3, 5], f32>, %b: !mo.tensor<[5, 5], f32>) -> !mo.tensor<[8, 5], f32> {
  %r = mo.concat[0] (%a, %b) : (!mo.tensor<[3, 5], f32>, !mo.tensor<[5, 5], f32>) -> !mo.tensor<[8, 5], f32>
  mo.output %r : !mo.tensor<[8, 5], f32>
}

// CHECK-LABEL: mo.graph @concat_outer_f32_rows_of_8
// CHECK:       map.iter.elementwise.body<[64]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[24], [40] -> [64], axis = 0>>
mo.graph @concat_outer_f32_rows_of_8(%a: !mo.tensor<[3, 8], f32>, %b: !mo.tensor<[5, 8], f32>) -> !mo.tensor<[8, 8], f32> {
  %r = mo.concat[0] (%a, %b) : (!mo.tensor<[3, 8], f32>, !mo.tensor<[5, 8], f32>) -> !mo.tensor<[8, 8], f32>
  mo.output %r : !mo.tensor<[8, 8], f32>
}

// CHECK-LABEL: mo.graph @concat_outer_bf16_rows_of_8
// CHECK:       map.iter.elementwise.body<[64]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[24], [40] -> [64], axis = 0>>
mo.graph @concat_outer_bf16_rows_of_8(%a: !mo.tensor<[3, 8], bf16>, %b: !mo.tensor<[5, 8], bf16>) -> !mo.tensor<[8, 8], bf16> {
  %r = mo.concat[0] (%a, %b) : (!mo.tensor<[3, 8], bf16>, !mo.tensor<[5, 8], bf16>) -> !mo.tensor<[8, 8], bf16>
  mo.output %r : !mo.tensor<[8, 8], bf16>
}

// A strided window (cut on dim 1) fused as a slice select into its
// producer's loop. Dim 1 is cut, so it can't merge with dim 0, and it can't
// start the innermost group either (a select keeps its innermost dim whole):
// only the two innermost dims merge, in the loop and in the window.
// CHECK-LABEL: mo.graph @slice_select
// CHECK:       map.iter.elementwise.body<[2, 4, 128]>
// CHECK:       map.index.slice.select<#map.index.slice.spec<[2, 4, 128] -> [2, 2, 128], starts = [0, 1, 0], stops = [2, 3, 128], steps = [1, 1, 1]>>
mo.graph @slice_select(%a: !mo.tensor<[2, 4, 8, 16], f32>, %b: !mo.tensor<[2, 4, 8, 16], f32>) -> (!mo.tensor<[2, 4, 8, 16], f32>, !mo.tensor<[2, 2, 8, 16], f32>) {
  %starts = mo.constant <#M.dense_array<0, 1, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %stops = mo.constant <#M.dense_array<2, 3, 8, 16> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %x = mo.add(%a, %b) : !mo.tensor<[2, 4, 8, 16], f32>
  %w = mo.slice(%x, %starts, %stops, %steps) : (!mo.tensor<[2, 4, 8, 16], f32>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[2, 2, 8, 16], f32>
  %y = mo.relu(%w) : !mo.tensor<[2, 2, 8, 16], f32>
  mo.output %x, %y : !mo.tensor<[2, 4, 8, 16], f32>, !mo.tensor<[2, 2, 8, 16], f32>
}

// A KV-cache-like concat along a dynamic axis: the dims after the axis (64)
// keep the merged extents' GCD a multiple of 16 whatever S is, so the axis
// merges with them.
// CHECK-LABEL: mo.graph @concat_dynamic_axis
// CHECK:       map.iter.elementwise.body<[B, [[P:param_flat_[0-9]+]]]>
// CHECK:       map.index.concat.select<#map.index.concat.spec<[B, 512], [B, {{param_flat_[0-9]+}}] -> [B, [[P]]], axis = 1>>
mo.graph @concat_dynamic_axis<B, S, T>(%cache: !mo.tensor<[B, 8, 64], f32>, %new: !mo.tensor<[B, S, 64], f32>) -> !mo.tensor<[B, T, 64], f32> {
  %r = mo.concat[1] (%cache, %new) : (!mo.tensor<[B, 8, 64], f32>, !mo.tensor<[B, S, 64], f32>) -> !mo.tensor<[B, T, 64], f32>
  mo.output %r : !mo.tensor<[B, T, 64], f32>
}

// A kernel that reads its index would see a different one after flattening:
// the loop is left alone.
// CHECK-LABEL: mo.graph @index_kernel
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.elementwise.body<[4, 6]>
mo.graph @index_kernel(%a: !mo.tensor<[4, 6], si32>) -> !mo.tensor<[4, 6], si32> {
  %y = mo.custom [symbol = "elemwise_add_with_index", device = #M.device_ref<"cpu", 0>](%a) : (!mo.tensor<[4, 6], si32>) -> !mo.tensor<[4, 6], si32>
  mo.output %y : !mo.tensor<[4, 6], si32>
}

// A broadcast whose input dim is symbolic replicates it or not depending on
// its run-time value, so its stride is unknown and neither neighbour merges.
// CHECK-LABEL: mo.graph @broadcast_unknown
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.elementwise.body<[B, S, 8]>
mo.graph @broadcast_unknown<B, S>(%a: !mo.tensor<[B, S, 8], f32>, %b: !mo.tensor<[S, 8], f32>) -> !mo.tensor<[B, S, 8], f32> {
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[S, 8], f32> -> !mo.tensor<[B, S, 8], f32>
  %y = mo.add(%a, %bb) : !mo.tensor<[B, S, 8], f32>
  mo.output %y : !mo.tensor<[B, S, 8], f32>
}

// A window of a broadcast: the two replicated dims keep stride 0 through
// the slice and merge; the innermost is real.
// CHECK-LABEL: mo.graph @slice_over_broadcast
// CHECK:       map.index.collapse.stride<#map.view.signature<[4, 4, 16] -> [16, 16]>, groups = [2, 1]>
// CHECK:       map.iter.elementwise.body<[16, 16]>
mo.graph @slice_over_broadcast(%b: !mo.tensor<[16], f32>) -> !mo.tensor<[4, 4, 16], f32> {
  %bb = mo.static.broadcast_to(%b) : !mo.tensor<[16], f32> -> !mo.tensor<[4, 8, 16], f32>
  %starts = mo.constant <#M.dense_array<0, 2, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %stops = mo.constant <#M.dense_array<4, 6, 16> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %steps = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %w = mo.slice(%bb, %starts, %stops, %steps) : (!mo.tensor<[4, 8, 16], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 16], f32>
  %y = mo.abs(%w) : !mo.tensor<[4, 4, 16], f32>
  mo.output %y : !mo.tensor<[4, 4, 16], f32>
}

// A reshape of a contiguous window: the reshape view is row-major over its
// output whatever it reads, so the loop flattens fully.
// CHECK-LABEL: mo.graph @reshape_over_slice
// CHECK:       map.index.collapse.stride<#map.view.signature<[4, 4, 4] -> [64]>, groups = [3]>
// CHECK:       map.iter.elementwise.body<[64]>
mo.graph @reshape_over_slice(%a: !mo.tensor<[8, 16], f32>) -> !mo.tensor<[4, 4, 4], f32> {
  %starts = mo.constant <#M.dense_array<2, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stops = mo.constant <#M.dense_array<6, 16> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %steps = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %w = mo.slice(%a, %starts, %stops, %steps) : (!mo.tensor<[8, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 16], f32>
  %r = mo.static.reshape(%w) : !mo.tensor<[4, 16], f32> -> !mo.tensor<[4, 4, 4], f32>
  %y = mo.abs(%r) : !mo.tensor<[4, 4, 4], f32>
  mo.output %y : !mo.tensor<[4, 4, 4], f32>
}

// Rank 4 with a transpose: only pairs the permutation keeps in order merge.
// Swapping the two innermost dims leaves the two outermost.
// CHECK-LABEL: mo.graph @transpose_inner_swap
// CHECK:       map.iter.elementwise.body<[6, 5, 4]>
mo.graph @transpose_inner_swap(%a: !mo.tensor<[2, 3, 4, 5], f32>) -> !mo.tensor<[2, 3, 5, 4], f32> {
  %perm = mo.constant <#M.dense_array<0, 1, 3, 2> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %t = mo.transpose(%a, %perm) : (!mo.tensor<[2, 3, 4, 5], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[2, 3, 5, 4], f32>
  %y = mo.abs(%t) : !mo.tensor<[2, 3, 5, 4], f32>
  mo.output %y : !mo.tensor<[2, 3, 5, 4], f32>
}

// Swapping the two middle dims breaks every pair.
// CHECK-LABEL: mo.graph @transpose_middle_swap
// CHECK-NOT:   map.index.collapse.stride
// CHECK:       map.iter.elementwise.body<[2, 4, 3, 5]>
mo.graph @transpose_middle_swap(%a: !mo.tensor<[2, 3, 4, 5], f32>) -> !mo.tensor<[2, 4, 3, 5], f32> {
  %perm = mo.constant <#M.dense_array<0, 2, 1, 3> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %t = mo.transpose(%a, %perm) : (!mo.tensor<[2, 3, 4, 5], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[2, 4, 3, 5], f32>
  %y = mo.abs(%t) : !mo.tensor<[2, 4, 3, 5], f32>
  mo.output %y : !mo.tensor<[2, 4, 3, 5], f32>
}

// In a control-flow region no parameter can be declared, so a symbolic pair
// stays as it is while static dims merge.
// CHECK-LABEL: mo.graph @in_if
// CHECK:       map.iter.elementwise.body<[T, 4]>
// CHECK:       map.index.collapse.stride<#map.view.signature<[4, 6] -> [24]>, groups = [2]>
// CHECK:       map.iter.elementwise.body<[24]>
mo.graph @in_if<T>(%a: !mo.tensor<[T, 4], f32>, %b: !mo.tensor<[4, 6], f32>, %cond: !mo.tensor<[], bool>) -> (!mo.tensor<[T, 4], f32>, !mo.tensor<[4, 6], f32>) {
  %x = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[T, 4], f32> {
    %r = mo.relu(%a) : !mo.tensor<[T, 4], f32>
    mo.yield %r : !mo.tensor<[T, 4], f32>
  } else {
    mo.yield %a : !mo.tensor<[T, 4], f32>
  }
  %y = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[4, 6], f32> {
    %r = mo.relu(%b) : !mo.tensor<[4, 6], f32>
    mo.yield %r : !mo.tensor<[4, 6], f32>
  } else {
    mo.yield %b : !mo.tensor<[4, 6], f32>
  }
  mo.output %x, %y : !mo.tensor<[T, 4], f32>, !mo.tensor<[4, 6], f32>
}

// A symbolic innermost extent merges like any other: the merged extent is the
// product of the pair, declared as a graph parameter.
// CHECK-LABEL: mo.graph @dynamic_innermost
// CHECK:       kgen.param.declare [[P:param_flat_[0-9]+]]: scalar<si64> = <mul_no_wrap(N, 4)>
// CHECK:       map.index.collapse.stride<#map.view.signature<[4, N] -> [[[P]]]>, groups = [2]>
// CHECK:       map.iter.elementwise.body<[[[P]]]>
mo.graph @dynamic_innermost<N>(%a: !mo.tensor<[4, N], f32>) -> !mo.tensor<[4, N], f32> {
  %y = mo.relu(%a) : !mo.tensor<[4, N], f32>
  mo.output %y : !mo.tensor<[4, N], f32>
}

// A group whose other extents are 1 is the named dim itself: no parameter is
// declared for it (an alias of a dim not yet known when MGP lowers the kernel
// would outlive its placeholder).
// CHECK-LABEL: mo.graph @alias_group
// CHECK-NOT:   kgen.param.declare
// CHECK:       map.iter.elementwise.body<[B]>
mo.graph @alias_group<B>(%a: !mo.tensor<[B, 1], f32>) -> !mo.tensor<[B, 1], f32> {
  %y = mo.relu(%a) : !mo.tensor<[B, 1], f32>
  mo.output %y : !mo.tensor<[B, 1], f32>
}
