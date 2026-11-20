// RUN: mo-opt %s --canonicalize --cse | FileCheck %s

// Test the folders for ops.

mo.graph @fold_split_scalar() -> (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>) {
    %num_splits = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
    %v1 = mo.constant <#M.dense_array<0.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
    %v2 = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %v3 = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>

    %o1 = mo.split(%v1, %num_splits) [axis = 0] : (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], f32>)
    %o2 = mo.split(%v2, %num_splits) [axis = 0] : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si64>)
    %o3 = mo.split(%v3, %num_splits) [axis = 0] : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si32>)

    // CHECK-LABEL: mo.graph @fold_split_scalar
    // CHECK: %[[CSE0:.*]] = mo.constant <#M.dense_array<0.000000e+00> : tensor<1xf32>> : !mo.tensor<[1], f32>
    // CHECK-NEXT: %[[CSE1:.*]] = mo.constant <#M.dense_array<2> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    // CHECK-NEXT: %[[CSE2:.*]] = mo.constant <#M.dense_array<1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
    // CHECK-NEXT: mo.output %[[CSE0]], %[[CSE1]], %[[CSE2]] : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
    mo.output %o1, %o2, %o3 : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
}

mo.graph @fold_split_scalar_non_consts(%arg0 : !mo.tensor<[1], f32>, %arg1 : !mo.tensor<[1], si64>, %arg2 : !mo.tensor<[1], si32>) -> (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>) {
    %num_splits = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[1], si64>

    %o1 = mo.split(%arg0, %num_splits) [axis = 0] : (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], f32>)
    %o2 = mo.split(%arg1, %num_splits) [axis = 0] : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si64>)
    %o3 = mo.split(%arg2, %num_splits) [axis = 0] : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si32>)

    // CHECK-LABEL: mo.graph @fold_split_scalar_non_consts
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[1], f32>, %[[ARG1:.*]]: !mo.tensor<[1], si64>, %[[ARG2:.*]]: !mo.tensor<[1], si32>)
    // CHECK-NEXT: mo.output %[[ARG0]], %[[ARG1]], %[[ARG2]] : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
    mo.output %o1, %o2, %o3 : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
}


mo.graph @fold_num_splits_1<N>(%arg0 : !mo.tensor<[100, N, 50], f32>) -> (!mo.tensor<[100, N, 50], f32>) {
    %num_splits = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[1], si64>
    %o1 = mo.split(%arg0, %num_splits) [axis = 0] : (!mo.tensor<[100, N, 50], f32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[100, N, 50], f32>)

    // CHECK-LABEL: mo.graph @fold_num_splits_1
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[100, N, 50], f32>)
    // CHECK-NEXT: mo.output %[[ARG0]]
    mo.output %o1 : !mo.tensor<[100, N, 50], f32>
}

mo.graph @fold_cast<N>(%arg0 : !mo.tensor<[100, N, 50], f32>) -> (!mo.tensor<[100, N, 50], f32>) {
    %output = mo.cast(%arg0) : (!mo.tensor<[100, N, 50], f32>) -> !mo.tensor<[100, N, 50],f32>
    // CHECK-LABEL: mo.graph @fold_cast
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[100, N, 50], f32>)
    // CHECK-NEXT: mo.output %[[ARG0]]
    mo.output %output : !mo.tensor<[100, N, 50], f32>
}

mo.graph @fold_cast_ui64_fp64<N>(%arg0 : !mo.tensor<[100, N, 50], f64>) -> (!mo.tensor<[100, N, 50], ui64>) {
    %output = mo.cast(%arg0) : (!mo.tensor<[100, N, 50], f64>) -> !mo.tensor<[100, N, 50],ui64>
    mo.output %output : !mo.tensor<[100, N, 50], ui64>
}
// CHECK-LABEL: mo.graph @fold_cast_ui64_fp64
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[100, N, 50], f64>)
// CHECK-NEXT: %[[OUT:.*]] = mo.cast(%[[ARG0]])
// CHECK-NEXT: mo.output %[[OUT]] : !mo.tensor<[100, N, 50], ui64>


mo.graph @fold_slice(%arg0: !mo.tensor<[5, 500], f64>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64> {
    %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[5, 500], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64>
    mo.output %2 : !mo.tensor<[5, 500], f64>
}
// CHECK-LABEL: mo.graph @fold_slice
// CHECK:(%[[ARG0:.*]]: !mo.tensor<[5, 500
// CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[5, 500], f64>


// There is some combination which should be able to produce the equiv of:
// np.arange(0, 15).reshape(3, 5)[::-1, ::-1]
// Which is the reverse with the same shape.
mo.graph @no_fold_slice_reverse(%arg0: !mo.tensor<[5, 500], f64>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64> {
    %step = mo.constant <#M.dense_array<-1, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[5, 500], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64>
    mo.output %2 : !mo.tensor<[5, 500], f64>
}
// CHECK-LABEL: mo.graph @no_fold_slice_reverse
// CHECK:(%[[ARG0:.*]]: !mo.tensor<[5, 500
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.slice
// CHECK-NEXT: mo.output %[[OUT]]


mo.graph @no_fold_slice_shape_mismatch(%arg0: !mo.tensor<[5, 500], f64>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[5, 250], f64> {
    %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[5, 500], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 250], f64>
    mo.output %2 : !mo.tensor<[5, 250], f64>
}
// CHECK-LABEL: mo.graph @no_fold_slice_shape_mismatch
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.slice
// CHECK-NEXT: mo.output %[[OUT]]


mo.graph @fold_slice_parameter<N, M>(%arg0: !mo.tensor<[N, M], f64>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[N, M], f64> {
    %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[N, M], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[N, M], f64>
    mo.output %2 : !mo.tensor<[N, M], f64>
}
// CHECK-LABEL: mo.graph @fold_slice_parameter<N, M>
// CHECK:(%[[ARG0:.*]]: !mo.tensor<[N, M
// CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[N, M], f64>


mo.graph @no_fold_slice_parameter<N, M>(%arg0: !mo.tensor<[N, M], f64>,
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[M, N], f64> {
    %step = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[N, M], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[M, N], f64>
    mo.output %2 : !mo.tensor<[M, N], f64>
}
// CHECK-LABEL: mo.graph @no_fold_slice_parameter<N, M>
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.slice
// CHECK-NEXT: mo.output %[[OUT]]


mo.graph @fold_reduce_add(%arg0 : !mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 10, 1], f32> {
    %res = mo.reduce.add(%arg0) : (!mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 10, 1], f32>

    // CHECK-LABEL: mo.graph @fold_reduce_add
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[1, 10, 1], f32>
    // CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[1, 10, 1], f32>
    mo.output %res : !mo.tensor<[1, 10, 1], f32>
}

mo.graph @fold_reduce_add2(%arg0 : !mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 10, 1], f32> {
    %res = mo.reduce.add(%arg0) : (!mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 10, 1], f32>

    // CHECK-LABEL: mo.graph @fold_reduce_add2
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[1, 10, 1], f32>
    // CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[1, 10, 1], f32>
    mo.output %res : !mo.tensor<[1, 10, 1], f32>
}

mo.graph @no_fold_reduce_add(%arg0 : !mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 1, 1], f32> {
    %res = mo.reduce.add(%arg0) : (!mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 1, 1], f32>

    // CHECK-LABEL: mo.graph @no_fold_reduce_add
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[1, 10, 1], f32>
    // CHECK-NEXT: %[[OUT:.*]] = mo.reduce.add
    // CHECK-NEXT: mo.output %[[OUT]] : !mo.tensor<[1, 1, 1], f32>
    mo.output %res : !mo.tensor<[1, 1, 1], f32>
}

mo.graph @no_fold_reduce_add2(%arg0 : !mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 1, 1], f32> {
    %res = mo.reduce.add(%arg0) : (!mo.tensor<[1, 10, 1], f32>) -> !mo.tensor<[1, 1, 1], f32>

    // CHECK-LABEL: mo.graph @no_fold_reduce_add2
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[1, 10, 1], f32>
    // CHECK-NEXT: %[[OUT:.*]] = mo.reduce.add
    // CHECK-NEXT: mo.output %[[OUT]] : !mo.tensor<[1, 1, 1], f32>
    mo.output %res : !mo.tensor<[1, 1, 1], f32>
}


mo.graph @fold_select_same(%cond : !mo.tensor<[5], bool>, %arg0 : !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32> {
    %res = mo.select(%cond, %arg0, %arg0) : (!mo.tensor<[5], bool>, !mo.tensor<[5], f32>, !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
    mo.output %res : !mo.tensor<[5], f32>
}
// CHECK-LABEL: mo.graph @fold_select_same
// CHECK: (%[[COND:.*]]: !mo.tensor<[5], bool>, %[[ARG0:.*]]: !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output %[[ARG0]]

mo.graph @fold_select_true(%arg0 : !mo.tensor<[5], f32>, %arg1 : !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32> {
    %cond = mo.constant <#M.dense_array<1, 1, 1, 1, 1> : !M.array<5xi8>> : !mo.tensor<[5], bool>
    %res = mo.select(%cond, %arg0, %arg1) : (!mo.tensor<[5], bool>, !mo.tensor<[5], f32>, !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
    mo.output %res : !mo.tensor<[5], f32>
}
// CHECK-LABEL: mo.graph @fold_select_true
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[5], f32>, %[[ARG1:.*]]: !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output %[[ARG0]]

mo.graph @fold_select_false(%arg0 : !mo.tensor<[5], f32>, %arg1 : !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32> {
    %cond = mo.constant <#M.dense_array<0, 0, 0, 0, 0> : !M.array<5xi8>> : !mo.tensor<[5], bool>
    %res = mo.select(%cond, %arg0, %arg1) : (!mo.tensor<[5], bool>, !mo.tensor<[5], f32>, !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
    mo.output %res : !mo.tensor<[5], f32>
}
// CHECK-LABEL: mo.graph @fold_select_false
// CHECK: (%[[ARG0:.*]]: !mo.tensor<[5], f32>, %[[ARG1:.*]]: !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output %[[ARG1]]

mo.graph @no_fold_select(%arg0 : !mo.tensor<[5], f32>, %arg1 : !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32> {
    %cond = mo.constant <#M.dense_array<1, 0, 1, 1, 1> : !M.array<5xi8>> : !mo.tensor<[5], bool>
    %res = mo.select(%cond, %arg0, %arg1) : (!mo.tensor<[5], bool>, !mo.tensor<[5], f32>, !mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
    mo.output %res : !mo.tensor<[5], f32>
}
// CHECK-LABEL: mo.graph @no_fold_select
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.select
// CHECK-NEXT: mo.output %[[OUT]]

// This is still folded as the axis has only one legal value.
mo.graph @fold_gather_dyn_axis() -> !mo.tensor<[3], si64> {
    %data = mo.constant <#M.dense_array<0, 11, 22, 33, 44, 55> : tensor<6xsi64>> : !mo.tensor<[6], si64>
    %indices = mo.constant <#M.dense_array<3, 1, 0> : tensor<3xsi32>> : !mo.tensor<[3], si32>
    %gather = mo.gather(%data, %indices) [axis = 0] : (!mo.tensor<[6], si64>, !mo.tensor<[3], si32>) -> !mo.tensor<[3], si64>
    mo.output %gather : !mo.tensor<[3], si64>
}
// CHECK-LABEL: mo.graph @fold_gather_dyn_axis
// CHECK-NEXT: %[[CST:.*]] = mo.constant <#M.dense_array<33, 11, 0>
// CHECK-NEXT: mo.output %[[CST]]

mo.graph @fold_concat(%axis : !mo.tensor<[], si64>) -> !mo.tensor<[4], si32> {
    %v1 = mo.constant <#M.dense_array<-1> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
    %v2 = mo.constant <#M.dense_array<-2> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
    %v3 = mo.constant <#M.dense_array<-3> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
    %v4 = mo.constant <#M.dense_array<-4> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
    %conc = mo.concat[-2] (%v1, %v2, %v3, %v4) : (!mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[4], si32>
    mo.output %conc : !mo.tensor<[4], si32>
}
// CHECK-LABEL: mo.graph @fold_concat
// CHECK-NEXT: %[[OUT:.*]] = mo.constant <#M.dense_array<-1, -2, -3, -4> : tensor<4xsi32>> : !mo.tensor<[4], si32>
// CHECK-NEXT: mo.output %[[OUT]]

mo.graph @fold_concat_multidim(%axis : !mo.tensor<[], si64>) -> !mo.tensor<[11], si32> {
    %v1 = mo.constant <#M.dense_array<-1> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
    %v2 = mo.constant <#M.dense_array<-2, -3, -4> : !M.array<3xsi32>> : !mo.tensor<[3], si32>
    %v3 = mo.constant <#M.dense_array<-5, -6, -7, -8, -9> : !M.array<5xsi32>> : !mo.tensor<[5], si32>
    %v4 = mo.constant <#M.dense_array<-10, -11> : !M.array<2xsi32>> : !mo.tensor<[2], si32>
    %conc = mo.concat[-2] (%v1, %v2, %v3, %v4) : (!mo.tensor<[1], si32>, !mo.tensor<[3], si32>, !mo.tensor<[5], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[11], si32>
    mo.output %conc : !mo.tensor<[11], si32>
}
// CHECK-LABEL: mo.graph @fold_concat_multidim
// CHECK-NEXT: %[[OUT:.*]] = mo.constant <#M.dense_array<-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11> : tensor<11xsi32>> : !mo.tensor<[11], si32>
// CHECK-NEXT: mo.output %[[OUT]]
mo.graph @no_fold_concat_dyn_in(%axis : !mo.tensor<[], si64>, %v4 : !mo.tensor<[2], si32>) -> !mo.tensor<[11], si32> {
    %v1 = mo.constant <#M.dense_array<-1> : !M.array<1xsi32>> : !mo.tensor<[1], si32>
    %v2 = mo.constant <#M.dense_array<-2, -3, -4> : !M.array<3xsi32>> : !mo.tensor<[3], si32>
    %v3 = mo.constant <#M.dense_array<-5, -6, -7, -8, -9> : !M.array<5xsi32>> : !mo.tensor<[5], si32>
    %conc = mo.concat[-2] (%v1, %v2, %v3, %v4) : (!mo.tensor<[1], si32>, !mo.tensor<[3], si32>, !mo.tensor<[5], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[11], si32>
    mo.output %conc : !mo.tensor<[11], si32>
}
// CHECK-LABEL: mo.graph @no_fold_concat_dyn_in
// CHECK: %[[OUT:.*]] = mo.concat
// CHECK-NEXT: mo.output %[[OUT]]

mo.graph @fold_concat_split(%in : !mo.tensor<[10, 2], f32>, %splitsizes: !mo.tensor<[5], si64>) -> !mo.tensor<[10, 2], f32> {
  %split:5 = mo.split(%in, %splitsizes) [axis = -2] : (!mo.tensor<[10, 2], f32>, !mo.tensor<[5], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %concat = mo.concat[-2] (%split#0, %split#1, %split#2, %split#3, %split#4) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[10, 2], f32>
  mo.output %concat : !mo.tensor<[10, 2], f32>
}
// Concat should fold then split should be dce'd
// CHECK-LABEL: mo.graph @fold_concat_split
// CHECK: mo.output

mo.graph @fold_concat_split_no_fold(%in : !mo.tensor<[10, 2], f32>, %splitsizes: !mo.tensor<[5], si64>) -> !mo.tensor<[10, 2], f32> {
  %split:5 = mo.split(%in, %splitsizes) [axis = -2] : (!mo.tensor<[10, 2], f32>, !mo.tensor<[5], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %concat = mo.concat[-2] (%split#1, %split#0, %split#2, %split#3, %split#4) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[10, 2], f32>
  mo.output %concat : !mo.tensor<[10, 2], f32>
}
// No folding as order isn't the same
// CHECK-LABEL: mo.graph @fold_concat_split_no_fold
// CHECK: mo.split
// CHECK-NEXT: %[[OUT:.*]] = mo.concat
// CHECK-NEXT: mo.output %[[OUT]]


mo.graph @fold_concat_split_no_fold_mismatch_elems(%in : !mo.tensor<[?, ?], f32>, %splitsizes: !mo.tensor<[6], si64>) -> !mo.tensor<[?, ?], f32> {
  %split:6 = mo.split(%in, %splitsizes) [axis = -2] : (!mo.tensor<[?, ?], f32>, !mo.tensor<[6], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %concat = mo.concat[-2] (%split#0, %split#1, %split#2, %split#3, %split#4) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %concat : !mo.tensor<[?, ?], f32>
}
// No folding as not all split outputs are accounted for
// CHECK-LABEL: mo.graph @fold_concat_split_no_fold_mismatch_elems
// CHECK: mo.split
// CHECK-NEXT: %[[OUT:.*]] = mo.concat
// CHECK-NEXT: mo.output %[[OUT]]

mo.graph @fold_concat_split_no_fold_uses_input(%in : !mo.tensor<[?, ?], f32>, %in2 : !mo.tensor<[2, 2], f32>, %splitsizes: !mo.tensor<[6], si64>) -> !mo.tensor<[?, ?], f32> {
  %split:6 = mo.split(%in, %splitsizes) [axis = -2] : (!mo.tensor<[?, ?], f32>, !mo.tensor<[6], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  %concat = mo.concat[-2] (%split#0, %split#1, %split#2, %split#3, %in2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %concat : !mo.tensor<[?, ?], f32>
}
// No folding as not all split outputs are accounted for
// CHECK-LABEL: mo.graph @fold_concat_split_no_fold_uses_input
// CHECK: mo.split
// CHECK-NEXT: %[[OUT:.*]] = mo.concat
// CHECK-NEXT: mo.output %[[OUT]]
