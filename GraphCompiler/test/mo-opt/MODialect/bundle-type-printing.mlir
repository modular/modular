// RUN: mo-opt %s --split-input-file | FileCheck %s

// Tests for the `!mo.bundle` pretty-printer and parser. The type is printed
// in a compact form when every element tensor has the same shape, dtype,
// device label, and metadata, and the device IDs are strictly ordered and
// contiguous. Otherwise the verbose `[<tensor>, <tensor>, ...]` form is used.

// Compact: multi-element bundle with contiguous GPU IDs.
// CHECK-LABEL: mo.graph @compact_multi
// CHECK-SAME: !mo.bundle<[3], f32, gpu:0-1>
mo.graph @compact_multi(
    %b: !mo.bundle<[3], f32, gpu:0-1>
) -> !mo.bundle<[3], f32, gpu:0-1> {
  mo.output %b
      : !mo.bundle<[3], f32, gpu:0-1>
}

// -----

// Compact: 4-way GPU range.
// CHECK-LABEL: mo.graph @compact_multi_four
// CHECK-SAME: !mo.bundle<[1024, 1024], bf16, gpu:0-3>
mo.graph @compact_multi_four(
    %b: !mo.bundle<[1024, 1024], bf16, gpu:0-3>
) -> !mo.bundle<[1024, 1024], bf16, gpu:0-3> {
  mo.output %b
      : !mo.bundle<[1024, 1024], bf16, gpu:0-3>
}

// -----

// Compact: single element with an explicit non-default device prints without
// a range suffix.
// CHECK-LABEL: mo.graph @compact_single_gpu
// CHECK-SAME: !mo.bundle<[3], f32, gpu:0>
// CHECK-NOT: gpu:0-0
mo.graph @compact_single_gpu(
    %b: !mo.bundle<[3], f32, gpu:0>
) -> !mo.bundle<[3], f32, gpu:0> {
  mo.output %b : !mo.bundle<[3], f32, gpu:0>
}

// -----

// Compact: single element on the default host device elides the device
// section entirely.
// CHECK-LABEL: mo.graph @compact_single_host
// CHECK-SAME: !mo.bundle<[3], f32>
mo.graph @compact_single_host(
    %b: !mo.bundle<[3], f32>
) -> !mo.bundle<[3], f32> {
  mo.output %b : !mo.bundle<[3], f32>
}

// -----

// Compact form parses as direct input (i.e. the user can write the compact
// form and it round-trips through mo-opt).
// CHECK-LABEL: mo.graph @compact_input
// CHECK-SAME: !mo.bundle<[5, 6], f32, gpu:2-4>
mo.graph @compact_input(
    %b: !mo.bundle<[5, 6], f32, gpu:2-4>
) -> !mo.bundle<[5, 6], f32, gpu:2-4> {
  mo.output %b : !mo.bundle<[5, 6], f32, gpu:2-4>
}

// -----

// Verbose fallback: element shapes differ.
// CHECK-LABEL: mo.graph @verbose_shape_mismatch
// CHECK-SAME: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>]>
mo.graph @verbose_shape_mismatch(
    %b: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>]>
) -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>]> {
  mo.output %b
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>]>
}

// -----

// Verbose fallback: element dtypes differ.
// CHECK-LABEL: mo.graph @verbose_dtype_mismatch
// CHECK-SAME: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], bf16, gpu:1>]>
mo.graph @verbose_dtype_mismatch(
    %b: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], bf16, gpu:1>]>
) -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], bf16, gpu:1>]> {
  mo.output %b
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], bf16, gpu:1>]>
}

// -----

// Verbose fallback: device labels differ (gpu + cpu). The second tensor uses
// `cpu:1` rather than `cpu:0` because `cpu:0` is the default host device and
// would be elided by the tensor printer, which is orthogonal to this test.
// CHECK-LABEL: mo.graph @verbose_label_mismatch
// CHECK-SAME: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, cpu:1>]>
mo.graph @verbose_label_mismatch(
    %b: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, cpu:1>]>
) -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, cpu:1>]> {
  mo.output %b
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, cpu:1>]>
}

// -----

// Verbose fallback: device IDs are not contiguous.
// CHECK-LABEL: mo.graph @verbose_noncontiguous
// CHECK-SAME: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:2>]>
mo.graph @verbose_noncontiguous(
    %b: !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:2>]>
) -> !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:2>]> {
  mo.output %b
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3], f32, gpu:2>]>
}

// -----

// Verbose fallback: device IDs are out of order.
// CHECK-LABEL: mo.graph @verbose_reversed
// CHECK-SAME: !mo.bundle<[!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>]>
mo.graph @verbose_reversed(
    %b: !mo.bundle<[!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>]>
) -> !mo.bundle<[!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>]> {
  mo.output %b
      : !mo.bundle<[!mo.tensor<[3], f32, gpu:1>, !mo.tensor<[3], f32, gpu:0>]>
}

// -----

// Compact form preserves a parameterized shape.
// CHECK-LABEL: mo.graph @compact_param_shape
// CHECK-SAME: !mo.bundle<[N, 16384], bf16, gpu:0-1>
mo.graph @compact_param_shape<N>(
    %b: !mo.bundle<[N, 16384], bf16, gpu:0-1>
) -> !mo.bundle<[N, 16384], bf16, gpu:0-1> {
  mo.output %b
      : !mo.bundle<[N, 16384], bf16, gpu:0-1>
}
