// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s --enable-var-scope


// CHECK:      mo.graph @bmm_no_rebind(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, 2], f32>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[2, 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32> {
mo.graph @bmm_no_rebind(
  %input: !mo.tensor<[4, 3, 2], f32>,
  %weight: !mo.tensor<[2, 3], f32>
) -> (!mo.tensor<[4, 3, 3], f32>) {

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[4, 3, 2], f32> -> !mo.tensor<[12, 2], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) : (!mo.tensor<[12, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[12, 3], f32>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[4, 2, 3], f32>
  %res = mo.batch_matmul(%input, %weight_bcast) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}


// CHECK:      mo.graph @bmm_with_rebind<[[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, [[D0]]], f32>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[[[D1]], 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32> {
mo.graph @bmm_with_rebind<D0, D1>(
  %input: !mo.tensor<[4, 3, D0], f32>,
  %weight: !mo.tensor<[D1, 3], f32>
) -> (!mo.tensor<[4, 3, 3], f32>) {

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[4, 3, [[D0]]], f32> -> !mo.tensor<[12, [[D0]]], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) : (!mo.tensor<[12, [[D0]]], f32>, !mo.tensor<[[[D1]], 3], f32>) -> !mo.tensor<[12, 3], f32>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[D1, 3], f32> -> !mo.tensor<[4, D1, 3], f32>
  %weight_bcast_rb = mo.rebind(%weight_bcast) : !mo.tensor<[4, D1, 3], f32> -> !mo.tensor<[4, D0, 3], f32>
  %res = mo.batch_matmul(%input, %weight_bcast_rb) : (!mo.tensor<[4, 3, D0], f32>, !mo.tensor<[4, D0, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}

// CHECK:      mo.graph @bmm_with_device(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, 2], f32, dev:0>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[2, 3], f32, dev:0>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32, dev:0> {
mo.graph @bmm_with_device(
  %input: !mo.tensor<[4, 3, 2], f32, dev:0>,
  %weight: !mo.tensor<[2, 3], f32, dev:0>
) -> (!mo.tensor<[4, 3, 3], f32, dev:0>){
  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[4, 3, 2], f32, dev:0> -> !mo.tensor<[12, 2], f32, dev:0>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) : (!mo.tensor<[12, 2], f32, dev:0>, !mo.tensor<[2, 3], f32, dev:0>) -> !mo.tensor<[12, 3], f32, dev:0>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[12, 3], f32, dev:0> -> !mo.tensor<[4, 3, 3], f32, dev:0>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[2, 3], f32, dev:0> -> !mo.tensor<[4, 2, 3], f32, dev:0>
  %res = mo.batch_matmul(%input, %weight_bcast): (!mo.tensor<[4, 3, 2], f32, dev:0>, !mo.tensor<[4, 2, 3], f32, dev:0>) -> !mo.tensor<[4, 3, 3], f32, dev:0>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32, dev:0>
  mo.output %res : !mo.tensor<[4, 3, 3], f32, dev:0>
}

// CHECK:      mo.graph @bmm_mixed_precision(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, 2], f16>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[2, 3], bf16>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32> {
mo.graph @bmm_mixed_precision(
  %input: !mo.tensor<[4, 3, 2], f16>,
  %weight: !mo.tensor<[2, 3], bf16>
) -> (!mo.tensor<[4, 3, 3], f32>) {

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[4, 3, 2], f16> -> !mo.tensor<[12, 2], f16>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) : (!mo.tensor<[12, 2], f16>, !mo.tensor<[2, 3], bf16>) -> !mo.tensor<[12, 3], f32>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[2, 3], bf16> -> !mo.tensor<[4, 2, 3], bf16>
  %res = mo.batch_matmul(%input, %weight_bcast) : (!mo.tensor<[4, 3, 2], f16>, !mo.tensor<[4, 2, 3], bf16>) -> !mo.tensor<[4, 3, 3], f32>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}

// CHECK:      mo.graph @bmm_batch_size_one(
// CHECK-SAME:   %[[A:.*]]: !mo.tensor<[1, 2, 3], f32>
// CHECK-SAME:   %[[B:.*]]: !mo.tensor<[1, 3, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[1, 2, 2], f32> {
mo.graph @bmm_batch_size_one(
  %A: !mo.tensor<[1, 2, 3], f32>,
  %B: !mo.tensor<[1, 3, 2], f32>
) -> (!mo.tensor<[1, 2, 2], f32>) {

  // CHECK-NEXT: %[[A_RESHAPE:.*]] = mo.static.reshape(%[[A]]) : !mo.tensor<[1, 2, 3], f32> -> !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: %[[B_RESHAPE:.*]] = mo.static.reshape(%[[B]]) : !mo.tensor<[1, 3, 2], f32> -> !mo.tensor<[3, 2], f32>
  // CHECK-NEXT: %[[MATMUL:.*]] = mo.matmul(%[[A_RESHAPE]], %[[B_RESHAPE]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[2, 2], f32>
  // CHECK-NEXT: %[[OUTPUT:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[1, 2, 2], f32>
  %res = mo.batch_matmul(%A, %B) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 3, 2], f32>) -> !mo.tensor<[1, 2, 2], f32>

  // CHECK-NEXT: mo.output %[[OUTPUT]] : !mo.tensor<[1, 2, 2], f32>
  mo.output %res : !mo.tensor<[1, 2, 2], f32>
}

// CHECK:      mo.graph @bmm_batch_size_ones(
// CHECK-SAME:   %[[A:.*]]: !mo.tensor<[1, 1, 1, 2, 3], f32>
// CHECK-SAME:   %[[B:.*]]: !mo.tensor<[1, 1, 1, 3, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[1, 1, 1, 2, 2], f32> {
mo.graph @bmm_batch_size_ones(
  %A: !mo.tensor<[1, 1, 1, 2, 3], f32>,
  %B: !mo.tensor<[1, 1, 1, 3, 2], f32>
) -> (!mo.tensor<[1, 1, 1, 2, 2], f32>) {

  // CHECK-NEXT: %[[A_RESHAPE:.*]] = mo.static.reshape(%[[A]]) : !mo.tensor<[1, 1, 1, 2, 3], f32> -> !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: %[[B_RESHAPE:.*]] = mo.static.reshape(%[[B]]) : !mo.tensor<[1, 1, 1, 3, 2], f32> -> !mo.tensor<[3, 2], f32>
  // CHECK-NEXT: %[[MATMUL:.*]] = mo.matmul(%[[A_RESHAPE]], %[[B_RESHAPE]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[2, 2], f32>
  // CHECK-NEXT: %[[OUTPUT:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[1, 1, 1, 2, 2], f32>
  %res = mo.batch_matmul(%A, %B) : (!mo.tensor<[1, 1, 1, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3, 2], f32>) -> !mo.tensor<[1, 1, 1, 2, 2], f32>

  // CHECK-NEXT: mo.output %[[OUTPUT]] : !mo.tensor<[1, 1, 1, 2, 2], f32>
  mo.output %res : !mo.tensor<[1, 1, 1, 2, 2], f32>
}

// CHECK:      mo.graph @bmm_batch_size_ones_mixed_precision(
// CHECK-SAME:   %[[A:.*]]: !mo.tensor<[1, 1, 1, 2, 3], ui8>
// CHECK-SAME:   %[[B:.*]]: !mo.tensor<[1, 1, 1, 3, 2], si8>
// CHECK-SAME: ) -> !mo.tensor<[1, 1, 1, 2, 2], si32> {
mo.graph @bmm_batch_size_ones_mixed_precision(
  %A: !mo.tensor<[1, 1, 1, 2, 3], ui8>,
  %B: !mo.tensor<[1, 1, 1, 3, 2], si8>
) -> (!mo.tensor<[1, 1, 1, 2, 2], si32>) {

  // CHECK-NEXT: %[[A_RESHAPE:.*]] = mo.static.reshape(%[[A]]) : !mo.tensor<[1, 1, 1, 2, 3], ui8> -> !mo.tensor<[2, 3], ui8>
  // CHECK-NEXT: %[[B_RESHAPE:.*]] = mo.static.reshape(%[[B]]) : !mo.tensor<[1, 1, 1, 3, 2], si8> -> !mo.tensor<[3, 2], si8>
  // CHECK-NEXT: %[[MATMUL:.*]] = mo.matmul(%[[A_RESHAPE]], %[[B_RESHAPE]]) : (!mo.tensor<[2, 3], ui8>, !mo.tensor<[3, 2], si8>) -> !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: %[[OUTPUT:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[2, 2], si32> -> !mo.tensor<[1, 1, 1, 2, 2], si32>
  %res = mo.batch_matmul(%A, %B) : (!mo.tensor<[1, 1, 1, 2, 3], ui8>, !mo.tensor<[1, 1, 1, 3, 2], si8>) -> !mo.tensor<[1, 1, 1, 2, 2], si32>

  // CHECK-NEXT: mo.output %[[OUTPUT]] : !mo.tensor<[1, 1, 1, 2, 2], si32>
  mo.output %res : !mo.tensor<[1, 1, 1, 2, 2], si32>
}

// CHECK-LABEL: @bmm_with_transpose
// CHECK: [[TRANSPOSE_RESULT:%.*]] = mo.transpose({{.*}}) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
// CHECK-NEXT: [[MATMUL_RESULT:%.*]] = mo.matmul({{%[0-9]+}}, [[TRANSPOSE_RESULT]]) : (!mo.tensor<[12, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[12, 3], f32>
// CHECK-NEXT: [[OUTPUT:%.*]] = mo.static.reshape([[MATMUL_RESULT]]) : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
// CHECK-NEXT: mo.output [[OUTPUT]] : !mo.tensor<[4, 3, 3], f32>
mo.graph @bmm_with_transpose(
  %input: !mo.tensor<[4, 3, 2], f32>,
  %weight: !mo.tensor<[3, 2], f32>
) -> (!mo.tensor<[4, 3, 3], f32>) {
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[3, 2], f32> -> !mo.tensor<[4, 3, 2], f32>
  %transpose = mo.transpose(%weight_bcast, %perm) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 3], f32>
  %res = mo.batch_matmul(%input, %transpose) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}

// CHECK-LABEL: @fail_bmm_not_const_perm
// CHECK-NOT: mo.matmul
// CHECK: mo.batch_matmul
mo.graph @fail_bmm_not_const_perm(
  %input: !mo.tensor<[4, 3, 2], f32>,
  %weight: !mo.tensor<[2, 3], f32>,
  %perm: !mo.tensor<[3], si64>
) -> (!mo.tensor<[4, 3, 3], f32>) {
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[4, 2, 3], f32>
  %transpose = mo.transpose(%weight_bcast, %perm) : (!mo.tensor<[4, 2, 3], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 2, 3], f32>
  %res = mo.batch_matmul(%input, %transpose) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}

// CHECK-LABEL: @fail_bmm_transpose_affects_batch_dim
// CHECK-NOT: mo.matmul
// CHECK: mo.batch_matmul
mo.graph @fail_bmm_transpose_affects_batch_dim(
  %input: !mo.tensor<[4, 4, 4], f32>,
  %weight: !mo.tensor<[4, 4], f32>
) -> (!mo.tensor<[4, 4, 4], f32>) {
  %perm = mo.constant <#M.dense_array<2, 0, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[4, 4], f32> -> !mo.tensor<[4, 4, 4], f32>
  %transpose = mo.transpose(%weight_bcast, %perm) : (!mo.tensor<[4, 4, 4], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 4, 4], f32>
  %res = mo.batch_matmul(%input, %transpose) : (!mo.tensor<[4, 4, 4], f32>, !mo.tensor<[4, 4, 4], f32>) -> !mo.tensor<[4, 4, 4], f32>
  mo.output %res : !mo.tensor<[4, 4, 4], f32>
}
