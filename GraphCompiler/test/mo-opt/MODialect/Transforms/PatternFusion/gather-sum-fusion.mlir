// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// Gather + reduce.add fusion: reduce.add(gather(x, idx) [axis = 0]) over axis 1
// fuses to mo.gather_sum when the gather axis is 0 and the reduce axis is 1.
// The fused op is single-device (MXF-821): a device-resident chain stays on
// its device instead of round-tripping through the host.

// CHECK-LABEL: mo.graph @gather_sum_fusion_cpu(
// CHECK: %[[RES:.*]] = "mo.gather_sum"(%arg0, %arg1) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3], f32>
// CHECK: mo.output %[[RES]] : !mo.tensor<[2, 1, 3], f32>
mo.graph @gather_sum_fusion_cpu(%table: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3], f32> {
  %gather = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 4, 3], f32>
  %res = mo.reduce.add(%gather) : (!mo.tensor<[2, 4, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %res : !mo.tensor<[2, 1, 3], f32>
}

// A device-resident chain fuses and keeps the input's device: no
// mo.transfer to the host is inserted.
// CHECK-LABEL: mo.graph @gather_sum_fusion_gpu(
// CHECK: %[[RES:.*]] = "mo.gather_sum"(%arg0, %arg1) : (!mo.tensor<[5, 3], f32, gpu:0>, !mo.tensor<[2, 4], si32, gpu:0>) -> !mo.tensor<[2, 1, 3], f32, gpu:0>
// CHECK: mo.output %[[RES]] : !mo.tensor<[2, 1, 3], f32, gpu:0>
mo.graph @gather_sum_fusion_gpu(%table: !mo.tensor<[5, 3], f32, gpu:0>, %indices: !mo.tensor<[2, 4], si32, gpu:0>) -> !mo.tensor<[2, 1, 3], f32, gpu:0> {
  %gather = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[5, 3], f32, gpu:0>, !mo.tensor<[2, 4], si32, gpu:0>) -> !mo.tensor<[2, 4, 3], f32, gpu:0>
  %res = mo.reduce.add(%gather) : (!mo.tensor<[2, 4, 3], f32, gpu:0>) -> !mo.tensor<[2, 1, 3], f32, gpu:0>
  mo.output %res : !mo.tensor<[2, 1, 3], f32, gpu:0>
}

// Wrong gather axis: no fusion.
// CHECK-LABEL: mo.graph @gather_wrong_axis_no_fuse(
// CHECK: mo.gather
// CHECK: mo.reduce.add
// CHECK-NOT: mo.gather_sum
mo.graph @gather_wrong_axis_no_fuse(%table: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[4, 1, 3], f32> {
  %gather = mo.gather(%table, %indices) [axis = 1] : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[4, 5, 3], f32>
  %res = mo.reduce.add(%gather) : (!mo.tensor<[4, 5, 3], f32>) -> !mo.tensor<[4, 1, 3], f32>
  mo.output %res : !mo.tensor<[4, 1, 3], f32>
}

// Wrong reduce axis: no fusion.
// CHECK-LABEL: mo.graph @reduce_wrong_axis_no_fuse(
// CHECK: mo.gather
// CHECK: mo.reduce.add
// CHECK-NOT: mo.gather_sum
mo.graph @reduce_wrong_axis_no_fuse(%table: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[1, 4, 3], f32> {
  %gather = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 4, 3], f32>
  %res = mo.reduce.add(%gather) : (!mo.tensor<[2, 4, 3], f32>) -> !mo.tensor<[1, 4, 3], f32>
  mo.output %res : !mo.tensor<[1, 4, 3], f32>
}

// int64 indices: the builtin kernel only supports int32, so the pair must
// stay unfused rather than fusing into an op the verifier rejects.
// CHECK-LABEL: mo.graph @gather_i64_indices_no_fuse(
// CHECK: mo.gather
// CHECK: mo.reduce.add
// CHECK-NOT: mo.gather_sum
mo.graph @gather_i64_indices_no_fuse(%table: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si64>) -> !mo.tensor<[2, 1, 3], f32> {
  %gather = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si64>) -> !mo.tensor<[2, 4, 3], f32>
  %res = mo.reduce.add(%gather) : (!mo.tensor<[2, 4, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %res : !mo.tensor<[2, 1, 3], f32>
}

// Rank-1 indices: the kernel strides the indices as [batch, multi_hot],
// so fusing would be a miscompile; the pair must stay unfused.
// CHECK-LABEL: mo.graph @gather_rank1_indices_no_fuse(
// CHECK: mo.gather
// CHECK: mo.reduce.add
// CHECK-NOT: mo.gather_sum
mo.graph @gather_rank1_indices_no_fuse(%table: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[4], si32>) -> !mo.tensor<[4, 1], f32> {
  %gather = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[5, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4, 3], f32>
  %res = mo.reduce.add(%gather) : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 1], f32>
  mo.output %res : !mo.tensor<[4, 1], f32>
}

// Rank-3 table: the kernel binds the table as [num_embeddings, emb_dim];
// a rank-3 gather result reducing to rank 4 has no fused equivalent.
// CHECK-LABEL: mo.graph @gather_rank3_table_no_fuse(
// CHECK: mo.gather
// CHECK: mo.reduce.add
// CHECK-NOT: mo.gather_sum
mo.graph @gather_rank3_table_no_fuse(%table: !mo.tensor<[2, 5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 5, 3], f32> {
  %gather = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[2, 5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 4, 5, 3], f32>
  %res = mo.reduce.add(%gather) : (!mo.tensor<[2, 4, 5, 3], f32>) -> !mo.tensor<[2, 1, 5, 3], f32>
  mo.output %res : !mo.tensor<[2, 1, 5, 3], f32>
}
