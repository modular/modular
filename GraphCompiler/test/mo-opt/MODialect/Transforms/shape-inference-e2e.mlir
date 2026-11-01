// UNSUPPORTED: ubsan

// Note: due to the size of the file, we want to filecheck with --enable-var-scope
// which prevents variables from being reused across CHECK-LABEL boundaries.
//
// RUN: mo-opt %s --mo-shape-inference --mo-hoist-param-exprs | FileCheck %s --enable-var-scope
// CHECK-NOT: kgen.func

mo.graph @static_ranks<D11, D12, D13, D14>(%arg0: !mo.tensor<[D11, D12], f32>, %arg1: !mo.tensor<[D13, D14], f32>) -> !mo.tensor<[D11, D12], f32> {
    %0 = mo.rebind(%arg1) : !mo.tensor<[D13, D14], f32> -> !mo.tensor<[D11, D12], f32>
    kgen.param.declare SORS_CH10: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[add] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> D11, D13), identical(:scalar<si64> D12, D14)))>
    %1 = mo.add(%arg0, %0) : !mo.tensor<[D11, D12], f32>
    %2 = mo.exp(%1) : !mo.tensor<[D11, D12], f32>
    %3 = mo.sub(%2, %1) : !mo.tensor<[D11, D12], f32>
    %4 = mo.relu(%3) : !mo.tensor<[D11, D12], f32>
    mo.output %4 : !mo.tensor<[D11, D12], f32>
}

// CHECK-LABEL: mo.graph
// CHECK-SAME: @static_ranks<[[D11:.*]], [[D12:.*]], [[D13:.*]], [[D14:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D11]], [[D12]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D13]], [[D14]]], f32>) -> !mo.tensor<[[[D11]], [[D12]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D13]], [[D14]]], f32> -> !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_CH10: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[add] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> [[D11]], [[D13]]), identical(:scalar<si64> [[D12]], [[D14]])))>
// CHECK-NEXT:   [[V1:%.*]] = mo.add([[VARG0]], [[V0]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.exp([[V1]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   [[V3:%.*]] = mo.sub([[V2]], [[V1]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   [[V4:%.*]] = mo.relu([[V3]]) : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[[[D11]], [[D12]]], f32>
// CHECK-NEXT: }

mo.graph @batch_matmul_static_rank<D1, D2, D3, D4, D5, D6>(%arg0: !mo.tensor<[D1, D2, D3], f32>, %arg1: !mo.tensor<[D4, D5, D6], f32>) -> !mo.tensor<[D1, D2, D6], f32> {
    kgen.param.declare assert20: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs batch dimensions must match" : !kgen.string}, identical(:scalar<si64> D1, D4))>
    kgen.param.declare assert22: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs inner dimensions must match" : !kgen.string}, identical(:scalar<si64> D3, D5))>
    %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6], f32>) -> !mo.tensor<[D1, D2, D6], f32>
    mo.output %0 : !mo.tensor<[D1, D2, D6], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @batch_matmul_static_rank<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]], [[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32> {
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs inner dimensions must match" : !kgen.string}, identical(:scalar<si64> [[D3]], [[D5]]))>
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs batch dimensions must match" : !kgen.string}, identical(:scalar<si64> [[D1]], [[D4]]))>
// CHECK-NEXT:   [[V1:%.*]] = mo.batch_matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], f32>, !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>) -> !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D1]], [[D2]], [[D6]]], f32>
// CHECK-NEXT: }

mo.graph @materialize_gather_axis_0_si32_e2e<D6, D7>(%arg0: !mo.tensor<[D6, 2], f32>, %arg1: !mo.tensor<[2, D7], si32>) -> !mo.tensor<[2, D7, 2], f32> {
    %1 = mo.gather(%arg0, %arg1) [axis = 0] : (!mo.tensor<[D6, 2], f32>, !mo.tensor<[2, D7], si32>) -> !mo.tensor<[2, D7, 2], f32>
    mo.output %1 : !mo.tensor<[2, D7, 2], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_gather_axis_0_si32_e2e<[[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D6]], 2], f32>, [[VARG1:%.*]]: !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[2, [[D7]], 2], f32> {
// CHECK-NEXT:   [[V1:%.*]] = mo.gather([[VARG0]], [[VARG1]]) [axis = 0] : (!mo.tensor<[[[D6]], 2], f32>, !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[2, [[D7]], 2], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[2, [[D7]], 2], f32>
// CHECK-NEXT: }

mo.graph @materialize_gather_axis_neg_1_si32_e2e<D6, D7>(%arg0: !mo.tensor<[D6, 2], f32>, %arg1: !mo.tensor<[2, D7], si32>) -> !mo.tensor<[D6, 2, D7], f32> {
    %1 = mo.gather(%arg0, %arg1) [axis = -1] : (!mo.tensor<[D6, 2], f32>, !mo.tensor<[2, D7], si32>) -> !mo.tensor<[D6, 2, D7], f32>
    mo.output %1 : !mo.tensor<[D6, 2, D7], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_gather_axis_neg_1_si32_e2e<[[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D6]], 2], f32>, [[VARG1:%.*]]: !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[[[D6]], 2, [[D7]]], f32> {
// CHECK-NEXT:   [[V1:%.*]] = mo.gather([[VARG0]], [[VARG1]]) [axis = -1] : (!mo.tensor<[[[D6]], 2], f32>, !mo.tensor<[2, [[D7]]], si32>) -> !mo.tensor<[[[D6]], 2, [[D7]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D6]], 2, [[D7]]], f32>
// CHECK-NEXT: }

mo.graph @materialize_mean_axis_0_si32_e2e<D4>(%arg0: !mo.tensor<[D4, 2], f32>) -> !mo.tensor<[1, 2], f32> {
    %sum = mo.reduce.add(%arg0) : (!mo.tensor<[D4, 2], f32>) -> !mo.tensor<[1, 2], f32>
    %len_idx = mosh.param.to_value = <D4>
    %len_u = mo.index.to_tensor(%len_idx) -> !mo.tensor<[], ui64>
    %len = mo.cast(%len_u) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
    %len_b = mo.static.broadcast_to(%len) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2], f32>
    %1 = mo.div(%sum, %len_b) : !mo.tensor<[1, 2], f32>
    mo.output %1 : !mo.tensor<[1, 2], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_mean_axis_0_si32_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[1, 2], f32> {
// CHECK-NEXT:   [[SUM:%.*]] = mo.reduce.add([[VARG0]]) : (!mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[1, 2], f32>
// CHECK-NEXT:   [[LEN_IDX:%.*]] = mosh.param.to_value = <[[D4]]>
// CHECK-NEXT:   [[LEN_U:%.*]] = mo.index.to_tensor([[LEN_IDX]]) -> !mo.tensor<[], ui64>
// CHECK-NEXT:   [[LEN:%.*]] = mo.cast([[LEN_U]]) : (!mo.tensor<[], ui64>) -> !mo.tensor<[], f32>
// CHECK-NEXT:   [[LEN_B:%.*]] = mo.static.broadcast_to([[LEN]]) : !mo.tensor<[], f32> -> !mo.tensor<[1, 2], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.div([[SUM]], [[LEN_B]]) : !mo.tensor<[1, 2], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[1, 2], f32>
// CHECK-NEXT: }

mo.graph @materialize_prod_axis_1_si64_e2e<D4>(%arg0: !mo.tensor<[D4, 2], f32>) -> !mo.tensor<[D4, 1], f32> {
    %1 = mo.reduce.mul(%arg0) [axis = 1] : (!mo.tensor<[D4, 2], f32>) -> !mo.tensor<[D4, 1], f32>
    mo.output %1 : !mo.tensor<[D4, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_prod_axis_1_si64_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32> {
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.mul([[VARG0]]) [axis = 1] : (!mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT: }

mo.graph @materialize_sum_axis_neg_1_si64_e2e<D4>(%arg0: !mo.tensor<[D4, 2], f32>) -> !mo.tensor<[D4, 1], f32> {
    %1 = mo.reduce.add(%arg0) : (!mo.tensor<[D4, 2], f32>) -> !mo.tensor<[D4, 1], f32>
    mo.output %1 : !mo.tensor<[D4, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @materialize_sum_axis_neg_1_si64_e2e<[[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32> {
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.add([[VARG0]]) : (!mo.tensor<[[[D4]], 2], f32>) -> !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D4]], 1], f32>
// CHECK-NEXT: }


mo.graph @broadcasted_binary_elwise<D21, D22, D23, D24 -> D27>(%arg0: !mo.tensor<[D21, 12, D22], f32>, %arg1: !mo.tensor<[D23, D24], f32>) -> !mo.tensor<[D21, 12, D27], f32> {
    kgen.param.declare assert33: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> D23, 1), identical(:scalar<si64> D23, 12)))>
    kgen.param.declare assert39: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> D22, D24), identical(:scalar<si64> D22, 1), identical(:scalar<si64> D24, 1)))>
    kgen.param.declare assert45: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 2 (D22) must be either 1 or equal to corresponding output dimension at index 2 (cond41)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D22, 1), D24, D22), D22), identical(:scalar<si64> D22, 1)))>
    %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[D21, 12, D22], f32> -> !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
    kgen.param.declare assert51: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 1 (D24) must be either 1 or equal to corresponding output dimension at index 2 (var47)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D22, 1), D24, D22), D24), identical(:scalar<si64> D24, 1)))>
    kgen.param.declare assert55: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 0 (D23) must be either 1 or equal to corresponding output dimension at index 1 (12)" : !kgen.string}, or(identical(:scalar<si64> D23, 1), identical(:scalar<si64> D23, 12)))>
    %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[D23, D24], f32> -> !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
    %2 = mo.add(%0, %1) : !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
    mo.output<cond(identical(:scalar<si64> D22, 1), D24, D22)> %2 : !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcasted_binary_elwise<[[D21:.*]], [[D22:.*]], [[D23:.*]], [[D24:.*]] -> [[D27:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D21]], 12, [[D22]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D23]], [[D24]]], f32>) -> !mo.tensor<[[[D21]], 12, [[D27]]], f32> {
// CHECK-DAG:   kgen.param.declare {{.*}}"mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> [[D23]], 1), identical(:scalar<si64> [[D23]], 12)))>
// CHECK-DAG:   kgen.param.declare {{.*}}"mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> [[D22]], [[D24]]), identical(:scalar<si64> [[D22]], 1), identical(:scalar<si64> [[D24]], 1)))>
// CHECK-DAG:   kgen.param.declare {{.*}}"mo.assert"{message = "[broadcast_to] input dimension at index 2 ([[D22]]) must be either 1 or equal to corresponding output dimension at index 2 ({{.*}})" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> [[D22]], 1), [[D24]], [[D22]]), [[D22]]), identical(:scalar<si64> [[D22]], 1)))>
// CHECK-DAG:    kgen.param.declare [[EXPR:.*]]: scalar<si64> = <cond(identical(:scalar<si64> [[D22]], 1), [[D24]], [[D22]])>
// CHECK-DAG:    [[V0:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D21]], 12, [[D22]]], f32> -> !mo.tensor<[[[D21]], 12, [[EXPR]]], f32>
// CHECK-DAG:    kgen.param.declare {{.*}} "mo.assert"{message = "[broadcast_to] input dimension at index 1 ([[D24]]) must be either 1 or equal to corresponding output dimension at index 2 ({{.*}})" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> [[D22]], 1), [[D24]], [[D22]]), [[D24]]), identical(:scalar<si64> [[D24]], 1)))>
// CHECK-DAG:    kgen.param.declare {{.*}} "mo.assert"{message = "[broadcast_to] input dimension at index 0 ([[D23]]) must be either 1 or equal to corresponding output dimension at index 1 (12)" : !kgen.string}, or(identical(:scalar<si64> [[D23]], 1), identical(:scalar<si64> [[D23]], 12)))>
// CHECK-DAG:    [[V1:%.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D23]], [[D24]]], f32> -> !mo.tensor<[[[D21]], 12, [[EXPR]]], f32>
// CHECK-NEXT:   [[V2:%.*]] = mo.add([[V0]], [[V1]]) : !mo.tensor<[[[D21]], 12, [[EXPR:.*]]], f32>
// CHECK-NEXT:   [[V3:%.*]] = mo.rebind([[V2]]) : !mo.tensor<[[[D21]], 12, [[EXPR]]], f32> -> !mo.tensor<[[[D21]], 12, cond(identical(:scalar<si64> [[D22]], 1), [[D24]], [[D22]])], f32>
// CHECK-NEXT:   mo.output<cond(identical(:scalar<si64> [[D22]], 1), [[D24]], [[D22]])> [[V3]] : !mo.tensor<[[[D21]], 12, cond(identical(:scalar<si64> [[D22]], 1), [[D24]], [[D22]])], f32>
// CHECK-NEXT: }

mo.graph @transpose_static_rank<D9, D10, D11, D12>(%arg0: !mo.tensor<[D9, D10], f32>, %arg1: !mo.tensor<[D11, D12], f32>) -> (!mo.tensor<[D9, D10], f32>, !mo.tensor<[D12, D11], f32>) {
    %0 = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %1 = mo.transpose(%arg1, %0) : (!mo.tensor<[D11, D12], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D12, D11], f32>
    mo.output %arg0, %1 : !mo.tensor<[D9, D10], f32>, !mo.tensor<[D12, D11], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @transpose_static_rank<[[D9:.*]], [[D10:.*]], [[D11:.*]], [[D12:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D9]], [[D10]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D11]], [[D12]]], f32>) -> (!mo.tensor<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D12]], [[D11]]], f32>) {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.transpose([[VARG1]], [[V0]]) : (!mo.tensor<[[[D11]], [[D12]]], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D12]], [[D11]]], f32>
// CHECK-NEXT:   mo.output [[VARG0]], [[V1]] : !mo.tensor<[[[D9]], [[D10]]], f32>, !mo.tensor<[[[D12]], [[D11]]], f32>
// CHECK-NEXT: }

mo.graph @scalar_to_tensor_reshapes_before_staticization<D18, D19>(%arg0: !mo.tensor<[D18, D19], si32>) -> !mo.tensor<[D18, D18], si32> {
    kgen.param.declare assert25: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[reshape] input and output number of elements must match: #mosh<ape[D18, D19]> : !mosh.ape != #mosh<ape[D18, D18]> : !mosh.ape\0AIf you are confident this is correct, consider a rebind to assert that the reshape is valid.\0AEx: ```\0A    n, m = x.shape\0A    x = x.rebind([(n // 4) * 4, m])\0A    x.reshape([n // 4, 4, m])\0A```" : !kgen.string}, identical(:scalar<si64> mul_no_wrap(D18, D18), mul_no_wrap(D18, D19)))>
    %0 = mo.rebind(%arg0) : !mo.tensor<[D18, D19], si32> -> !mo.tensor<[D18, D18], si32>
    mo.output %0 : !mo.tensor<[D18, D18], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @scalar_to_tensor_reshapes_before_staticization<[[D18:.*]], [[D19:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D18]], [[D19]]], si32>) -> !mo.tensor<[[[D18]], [[D18]]], si32> {
// CHECK-NEXT: kgen.param.declare [[P29:.*]]: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[reshape] input and output number of elements must match: #mosh<ape[D18, D19]> : !mosh.ape != #mosh<ape[D18, D18]> : !mosh.ape[[REST:.*]]" : !kgen.string}, identical(:scalar<si64> mul_no_wrap(D18, D18), mul_no_wrap(D18, D19)))>
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG0]]) : !mo.tensor<[[[D18]], [[D19]]], si32> -> !mo.tensor<[[[D18]], [[D18]]], si32>
// CHECK-NEXT:   mo.output [[V0]] : !mo.tensor<[[[D18]], [[D18]]], si32>
// CHECK-NEXT: }

mo.graph @binary_op_asserts<D13, D14, D15, D16>(%arg0: !mo.tensor<[D13, D14], f32>, %arg1: !mo.tensor<[D15, D16], f32>) -> !mo.tensor<[D13, D14], f32> {
    %0 = mo.add(%arg0, %arg0) : !mo.tensor<[D13, D14], f32>
    %1 = mo.rebind(%arg1) : !mo.tensor<[D15, D16], f32> -> !mo.tensor<[D13, D14], f32>
    kgen.param.declare SORS_CH12: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[add] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> D13, D15), identical(:scalar<si64> D14, D16)))>
    %2 = mo.add(%arg0, %1) : !mo.tensor<[D13, D14], f32>
    %3 = mo.add(%0, %2) : !mo.tensor<[D13, D14], f32>
    mo.output %3 : !mo.tensor<[D13, D14], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_op_asserts<[[D13:.*]], [[D14:.*]], [[D15:.*]], [[D16:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D13]], [[D14]]], f32>, [[VARG1:%.*]]: !mo.tensor<[[[D15]], [[D16]]], f32>) -> !mo.tensor<[[[D13]], [[D14]]], f32> {
// CHECK:        [[V0:%.*]] = mo.add([[VARG0]], {{.*}}) : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D15]], [[D16]]], f32> -> !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   kgen.param.declare SORS_CH12: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[add] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> [[D13]], [[D15]]), identical(:scalar<si64> [[D14]], [[D16]])))>
// CHECK-NEXT:   [[V2:%.*]] = mo.add([[VARG0]], [[V1]]) : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   [[V3:%.*]] = mo.add([[V0]], [[V2]]) : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[[[D13]], [[D14]]], f32>
// CHECK-NEXT: }

mo.graph @binary_op_no_assert<D7, D8>(%arg0: !mo.tensor<[D7, D8], f32>) -> !mo.tensor<[D7, D8], f32> {
    %0 = mo.add(%arg0, %arg0) : !mo.tensor<[D7, D8], f32>
    %1 = mo.add(%arg0, %0) : !mo.tensor<[D7, D8], f32>
    mo.output %1 : !mo.tensor<[D7, D8], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_op_no_assert<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], [[D8]]], f32>) -> !mo.tensor<[[[D7]], [[D8]]], f32> {
// CHECK:        [[V0:%.*]] = mo.add([[VARG0]], {{.*}}) : !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT:   [[V1:%.*]] = mo.add([[VARG0]], [[V0]]) : !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D7]], [[D8]]], f32>
// CHECK-NEXT: }

mo.graph @batch_matmul_reused_rhs<D0, D1 -> D2, D3, D4>(
  %arg0: !mo.tensor<[3, D0, 4], f32>,
  %arg1: !mo.tensor<[3, 4, D1], f32>
) -> !mo.tensor<[D2, D3, D4], f32> {
  %bmm1 = mo.batch_matmul<() -> D5, D6, D7>(%arg0, %arg1) : (!mo.tensor<[3, D0, 4], f32>, !mo.tensor<[3, 4, D1], f32>) -> !mo.tensor<[D5, D6, D7], f32>
  %bmm2 = mo.batch_matmul<() -> D8, D9, D10>(%bmm1, %arg1) : (!mo.tensor<[D5, D6, D7], f32>, !mo.tensor<[3, 4, D1], f32>) -> !mo.tensor<[D8, D9, D10], f32>
  mo.output<D8, D9, D10> %bmm2 : !mo.tensor<[D8, D9, D10], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @batch_matmul_reused_rhs<[[D0:.*]], [[D1:.*]]>([[VARG0:%.*]]: !mo.tensor<[3, [[D0]], 4], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul([[VARG0]], [[VARG1]]) : (!mo.tensor<[3, [[D0]], 4], f32>, !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   kgen.param.declare [[P25:.*]]: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[batch_matmul] inputs inner dimensions must match" : !kgen.string}, identical(:scalar<si64> [[D1]], 4))>
// CHECK-NEXT:   [[V2:%.*]] = mo.batch_matmul([[V0]], [[VARG1]]) : (!mo.tensor<[3, [[D0]], [[D1]]], f32>, !mo.tensor<[3, 4, [[D1]]], f32>) -> !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT:   mo.output [[V2]] : !mo.tensor<[3, [[D0]], [[D1]]], f32>
// CHECK-NEXT: }


mo.graph @slice_neg_stop(%arg0: !mo.tensor<[10], f32>) -> !mo.tensor<[3], f32> {
    %0 = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
    %1 = mo.constant <#M.dense_array<-1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
    %2 = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
    %3 = mo.slice(%arg0, %0, %1, %2) : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
    mo.output %3 : !mo.tensor<[3], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_neg_stop([[VARG0:%.*]]: !mo.tensor<[10], f32>) -> !mo.tensor<[3], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<3> : tensor<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<-1> : tensor<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.slice([[VARG0]], [[V0]], [[V1]], [[V2]]) : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[3], f32>
// CHECK-NEXT: }

mo.graph @slice_clamped_start_stop_neg_step(%arg0: !mo.tensor<[10], f32>) -> !mo.tensor<[5], f32> {
    %0 = mo.constant <#M.dense_array<30> : tensor<1xsi32>> : !mo.tensor<[1], si32>
    %1 = mo.constant <#M.dense_array<-20> : tensor<1xsi32>> : !mo.tensor<[1], si32>
    %2 = mo.constant <#M.dense_array<-2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
    %3 = mo.slice(%arg0, %0, %1, %2) : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
    mo.output %3 : !mo.tensor<[5], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_clamped_start_stop_neg_step([[VARG0:%.*]]: !mo.tensor<[10], f32>) -> !mo.tensor<[5], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<30> : tensor<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<-20> : tensor<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<-2> : tensor<1xsi32>> : !mo.tensor<[1], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.slice([[VARG0]], [[V0]], [[V1]], [[V2]]) : (!mo.tensor<[10], f32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[5], f32>
// CHECK-NEXT: }

mo.graph @slice_4d(%arg0: !mo.tensor<[10, 20, 30, 40], f32>) -> !mo.tensor<[3, 3, 8, 8], f32> {
    %0 = mo.constant <#M.dense_array<3, -2, -100, 200> : tensor<4xsi32>> : !mo.tensor<[4], si32>
    %1 = mo.constant <#M.dense_array<-1, 10, 200, -100> : tensor<4xsi32>> : !mo.tensor<[4], si32>
    %2 = mo.constant <#M.dense_array<2, -3, 4, -5> : tensor<4xsi32>> : !mo.tensor<[4], si32>
    %3 = mo.slice(%arg0, %0, %1, %2) : (!mo.tensor<[10, 20, 30, 40], f32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[3, 3, 8, 8], f32>
    mo.output %3 : !mo.tensor<[3, 3, 8, 8], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_4d([[VARG0:%.*]]: !mo.tensor<[10, 20, 30, 40], f32>) -> !mo.tensor<[3, 3, 8, 8], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<3, -2, -100, 200> : tensor<4xsi32>> : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<-1, 10, 200, -100> : tensor<4xsi32>> : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<2, -3, 4, -5> : tensor<4xsi32>> : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V3:%.*]] = mo.slice([[VARG0]], [[V0]], [[V1]], [[V2]]) : (!mo.tensor<[10, 20, 30, 40], f32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[3, 3, 8, 8], f32>
// CHECK-NEXT:   mo.output [[V3]] : !mo.tensor<[3, 3, 8, 8], f32>
// CHECK-NEXT: }

// CHECK-LABEL: mo.graph @scatter
// CHECK-SAME:  <[[N:.*]]>(
// CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[N, 2, 3], f32>,
// CHECK-SAME:    %[[UPDATES:.*]]: !mo.tensor<[1, 1, 2], f32>,
// CHECK-SAME:    %[[INDICES:.*]]: !mo.tensor<[1, 1, 2], si32>
mo.graph @scatter<N>(%arg0: !mo.tensor<[N, 2, 3], f32>, %arg1: !mo.tensor<[1, 1, 2], f32>, %arg2: !mo.tensor<[1, 1, 2], si32>) -> !mo.tensor<[N, 2, 3], f32> {
    kgen.param.declare assert2: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[scatter] indices shape cannot be bigger than input shape" : !kgen.string}, ge(:scalar<si64> N, 1))>
    %1 = mo.scatter(%arg0, %arg1, %arg2) [axis = -1] : (!mo.tensor<[N, 2, 3], f32>, !mo.tensor<[1, 1, 2], f32>, !mo.tensor<[1, 1, 2], si32>) -> !mo.tensor<[N, 2, 3], f32>
    mo.output %1 : !mo.tensor<[N, 2, 3], f32>
}

mo.graph @scatter_nd(%arg0: !mo.tensor<[10, 1, 1], f32>, %arg1: !mo.tensor<[5, 1, 1], f32>, %arg2: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> {
    %0 = mo.static.reshape(%arg2) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
    %1 = mo.scatter_nd(%arg0, %arg1, %0) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
    mo.output %1 : !mo.tensor<[10, 1, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_nd([[VARG0:[^ ]*]]: !mo.tensor<[10, 1, 1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 1, 1], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG2]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter_nd([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[10, 1, 1], f32>

mo.graph @scatter_nd_add(%arg0: !mo.tensor<[10, 1, 1], f32>, %arg1: !mo.tensor<[5, 1, 1], f32>, %arg2: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> {
    %0 = mo.static.reshape(%arg2) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
    %1 = mo.scatter_nd.add(%arg0, %arg1, %0) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
    mo.output %1 : !mo.tensor<[10, 1, 1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @scatter_nd_add([[VARG0:[^ ]*]]: !mo.tensor<[10, 1, 1], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[5, 1, 1], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[10, 1, 1], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.static.reshape([[VARG2]]) : !mo.tensor<[5], si32> -> !mo.tensor<[5, 1], si32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.scatter_nd.add([[VARG0]], [[VARG1]], [[V0]]) : (!mo.tensor<[10, 1, 1], f32>, !mo.tensor<[5, 1, 1], f32>, !mo.tensor<[5, 1], si32>) -> !mo.tensor<[10, 1, 1], f32>
// CHECK-NEXT: mo.output [[V1]] : !mo.tensor<[10, 1, 1], f32>

mo.graph @conv2d_static_input_dims(%arg0: !mo.tensor<[10, 5, 5, 16], f32>, %arg1: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 2, 2, 32], f32> {
    %0 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %1 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %3 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
    %4 = mo.conv(%arg0, %arg1) [strides = %0, dilations = %1, paddings = %2, num_groups = %3, filter_layout = ""] : (
    !mo.tensor<[10, 5, 5, 16], f32>,
    !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[10, 2, 2, 32], f32>
    mo.output %4 : !mo.tensor<[10, 2, 2, 32], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @conv2d_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 5, 5, 16], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 2, 2, 32], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V0]], dilations = [[V1]], paddings = [[V2]], num_groups = [[V3]], filter_layout = ""] : (!mo.tensor<[10, 5, 5, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[10, 2, 2, 32], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[10, 2, 2, 32], f32>
// CHECK-NEXT: }

mo.graph @conv2d_dyn_input_dims<D6>(%arg0: !mo.tensor<[D6, 20, 30, 16], f32>, %arg1: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[D6, 7, 12, 32], f32> {
    %0 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
    %1 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %2 = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %3 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %4 = mo.conv(%arg0, %arg1) [strides = %3, dilations = %2, paddings = %1, num_groups = %0, filter_layout = ""] : (
    !mo.tensor<[D6, 20, 30, 16], f32>,
    !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[], si64>
  ) -> !mo.tensor<[D6, 7, 12, 32], f32>
    mo.output %4 : !mo.tensor<[D6, 7, 12, 32], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @conv2d_dyn_input_dims<[[D9:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D9]], 20, 30, 16], f32>, [[VARG1:%.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[[[D9]], 7, 12, 32], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V3]], dilations = [[V2]], paddings = [[V1]], num_groups = [[V0]], filter_layout = ""] : (!mo.tensor<[[[D9]], 20, 30, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D9]], 7, 12, 32], f32>
// CHECK-NEXT:   mo.output [[V4]]
// CHECK-NEXT: }

// NOTE this is essentially the inverse of `conv2d_dyn_input_dims`, although we
// can't get the result shape to match the input shape in
// `conv2d_dyn_input_dims` due to roundings.
// CHECK-LABEL: mo.graph @conv_transpose_2d_dyn_input_dims
//  CHECK-SAME:   <[[D9:.*]]>(
mo.graph @conv_transpose_2d_dyn_input_dims<D6>(%arg0: !mo.tensor<[D6, 7, 12, 32], f32>, %arg1: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[D6, 19, 29, 16], f32> {
    %0 = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %1 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %2 = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %3 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %4 = mo.conv_transpose(%arg0, %arg1) [strides = %3, dilations = %2, paddings = %1, output_paddings = %0, filter_layout = ""] : (
    !mo.tensor<[D6, 7, 12, 32], f32>,
    !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>,
    !mo.tensor<[2], si64>
  ) -> !mo.tensor<[D6, 19, 29, 16], f32>
    mo.output %4 : !mo.tensor<[D6, 19, 29, 16], f32>
}

mo.graph @max_pool2d_static_input_dims(%arg0: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 1, 1, 16], f32> {
    %0 = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %3 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %4 = mo.max_pool(%arg0) [filter_shape = %0, strides = %1, dilations = %2, paddings = %3] : (
    !mo.tensor<[10, 4, 4, 16], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[10, 1, 1, 16], f32>
    mo.output %4 : !mo.tensor<[10, 1, 1, 16], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @max_pool2d_static_input_dims([[VARG0:%.*]]: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 1, 1, 16], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.max_pool([[VARG0]]) [filter_shape = [[V0]], strides = [[V1]], dilations = [[V2]], paddings = [[V3]]] : (!mo.tensor<[10, 4, 4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[10, 1, 1, 16], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[10, 1, 1, 16], f32>
// CHECK-NEXT: }

mo.graph @max_pool2d_ceil_mode_true_static_input_dims(%arg0: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 2, 2, 16], f32> {
    %0 = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %1 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %3 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %4 = mo.max_pool_ceil_mode_true(%arg0) [filter_shape = %0, strides = %1, dilations = %2, paddings = %3] : (
    !mo.tensor<[10, 4, 4, 16], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[10, 2, 2, 16], f32>
    mo.output %4 : !mo.tensor<[10, 2, 2, 16], f32>
}
// CHECK-LABEL: mo.graph @max_pool2d_ceil_mode_true_static_input_dims
// CHECK-SAME: ([[VARG0:%.*]]: !mo.tensor<[10, 4, 4, 16], f32>) -> !mo.tensor<[10, 2, 2, 16], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.max_pool_ceil_mode_true([[VARG0]]) [filter_shape = [[V0]], strides = [[V1]], dilations = [[V2]], paddings = [[V3]]] : (!mo.tensor<[10, 4, 4, 16], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[10, 2, 2, 16], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[10, 2, 2, 16], f32>
// CHECK-NEXT: }

mo.graph @avg_pool2d_dyn_input_dims<D7, D8>(%arg0: !mo.tensor<[D7, 20, 30, D8], f32>) -> !mo.tensor<[D7, 9, 14, D8], f32> {
    %0 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %1 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %3 = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %4 = mo.avg_pool(%arg0) [filter_shape = %3, strides = %2, dilations = %1, paddings = %0, count_boundary = false] : (
    !mo.tensor<[D7, 20, 30, D8], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[D7, 9, 14, D8], f32>
    mo.output %4 : !mo.tensor<[D7, 9, 14, D8], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @avg_pool2d_dyn_input_dims<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>) -> !mo.tensor<[[[D7]], 9, 14, [[D8]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.avg_pool([[VARG0]]) [filter_shape = [[V3]], strides = [[V2]], dilations = [[V1]], paddings = [[V0]], count_boundary = false] : (!mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D7]], 9, 14, [[D8]]], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[[[D7]], 9, 14, [[D8]]], f32>
// CHECK-NEXT: }

mo.graph @avg_pool2d_ceil_mode_true_dyn_input_dims<D7, D8>(%arg0: !mo.tensor<[D7, 20, 30, D8], f32>) -> !mo.tensor<[D7, 10, 15, D8], f32> {
    %0 = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
    %1 = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2 = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %3 = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %4 = mo.avg_pool_ceil_mode_true(%arg0) [filter_shape = %3, strides = %2, dilations = %1, paddings = %0, count_boundary = false] : (
    !mo.tensor<[D7, 20, 30, D8], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[D7, 10, 15, D8], f32>
    mo.output %4 : !mo.tensor<[D7, 10, 15, D8], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @avg_pool2d_ceil_mode_true_dyn_input_dims<[[D7:.*]], [[D8:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>) -> !mo.tensor<[[[D7]], 10, 15, [[D8]]], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.constant <#M.dense_array<3, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.avg_pool_ceil_mode_true([[VARG0]]) [filter_shape = [[V3]], strides = [[V2]], dilations = [[V1]], paddings = [[V0]], count_boundary = false] : (!mo.tensor<[[[D7]], 20, 30, [[D8]]], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D7]], 10, 15, [[D8]]], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[[[D7]], 10, 15, [[D8]]], f32>
// CHECK-NEXT: }

mo.graph @arg_min<D5, D6>(%arg0: !mo.tensor<[D5, D6], si64>) -> !mo.tensor<[D5, 1], si64> {
    %1 = mo.reduce.arg_min(%arg0) [axis = 1] : (!mo.tensor<[D5, D6], si64>) -> !mo.tensor<[D5, 1], si64>
    mo.output %1 : !mo.tensor<[D5, 1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @arg_min<[[D5:.*]], [[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D5]], [[D6]]], si64>) -> !mo.tensor<[[[D5]], 1], si64> {
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.arg_min([[VARG0]]) [axis = 1] : (!mo.tensor<[[[D5]], [[D6]]], si64>) -> !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT: }


mo.graph @arg_max<D5, D6>(%arg0: !mo.tensor<[D5, D6], si64>) -> !mo.tensor<[D5, 1], si64> {
    %1 = mo.reduce.arg_max(%arg0) [axis = 1] : (!mo.tensor<[D5, D6], si64>) -> !mo.tensor<[D5, 1], si64>
    mo.output %1 : !mo.tensor<[D5, 1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @arg_max<[[D5:.*]], [[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D5]], [[D6]]], si64>) -> !mo.tensor<[[[D5]], 1], si64> {
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.arg_max([[VARG0]]) [axis = 1] : (!mo.tensor<[[[D5]], [[D6]]], si64>) -> !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[[[D5]], 1], si64>
// CHECK-NEXT: }

// This test ensures we can shape infer through a `mo.concat` op with non-equal
// dimensions (i.e., it doesn't rely on shape inference being able to prove the
// dimensions are equal). This is important because sometimes we can't prove as
// much as we want, but we still want to shape infer through `mo.concat`, which
// is an important op for shape inference in models like transformer.
mo.graph @concat_non_equal_dims<D10, D11, D12, D13, D14, D15>(%arg0: !mo.tensor<[D10, 1, D11], f32>, %arg1: !mo.tensor<[D12, 2, D13], f32>, %arg2: !mo.tensor<[D14, 3, D15], f32>) -> !mo.tensor<[D10, 6, D11], f32> {
    %0 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
    kgen.param.declare assert20: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:!kgen.scalar<si64> D10, D12))>
    kgen.param.declare assert22: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:!kgen.scalar<si64> D11, D13))>
    kgen.param.declare assert24: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:!kgen.scalar<si64> D10, D14))>
    kgen.param.declare assert26: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[concat] input shapes must match except at concat axis" : !kgen.string}, eq(:!kgen.scalar<si64> D11, D15))>
    %1 = mo.concat[1] (%arg0, %arg1, %arg2) : (!mo.tensor<[D10, 1, D11], f32>, !mo.tensor<[D12, 2, D13], f32>, !mo.tensor<[D14, 3, D15], f32>) -> !mo.tensor<[D10, 6, D11], f32>
    mo.output %1 : !mo.tensor<[D10, 6, D11], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @concat_non_equal_dims<[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]], [[D5:.*]]>([[INPUT1:%.*]]: !mo.tensor<[[[D0]], 1, [[D1]]], f32>, [[INPUT2:%.*]]: !mo.tensor<[[[D2]], 2, [[D3]]], f32>, [[INPUT3:%.*]]: !mo.tensor<[[[D4]], 3, [[D5]]], f32>) -> !mo.tensor<[[[D0]], 6, [[D1]]], f32> {
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} identical(:scalar<si64> [[D0]], [[D2]])
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} identical(:scalar<si64> [[D1]], [[D3]])
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} identical(:scalar<si64> [[D0]], [[D4]])
// CHECK-NEXT:   {{.*}}"mo.assert"{{.*}} "[concat] input shapes must match except at concat axis" {{.*}} identical(:scalar<si64> [[D1]], [[D5]])
// CHECK-NEXT:   [[RES:%.*]] = mo.concat{{.*}}([[INPUT1]], [[INPUT2]], [[INPUT3]]) : {{.*}} -> !mo.tensor<[[[D0]], 6, [[D1]]], f32>
// CHECK-NEXT:   mo.output [[RES]]
// CHECK-NEXT: }

mo.graph @pad_constant(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], f32>) -> !mo.tensor<[32, 73], f32> {
    %0 = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
    %1 = mo.pad.constant(%arg0, %0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[32, 73], f32>
    mo.output %1 : !mo.tensor<[32, 73], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pad_constant([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>, [[VARG1:%.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[32, 73], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.pad.constant([[VARG0]], [[V0]], [[VARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[32, 73], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[32, 73], f32>
// CHECK-NEXT: }

mo.graph @pad_reflect(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> {
    %0 = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
    %1 = mo.pad.reflect(%arg0, %0) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[32, 73], f32>
    mo.output %1 : !mo.tensor<[32, 73], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pad_reflect([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.pad.reflect([[VARG0]], [[V0]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[32, 73], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[32, 73], f32>
// CHECK-NEXT: }

mo.graph @pad_repeat(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> {
    %0 = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
    %1 = mo.pad.repeat(%arg0, %0) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[32, 73], f32>
    mo.output %1 : !mo.tensor<[32, 73], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @pad_repeat([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[32, 73], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<10, 20, 30, 40> : tensor<4xsi32>> : !mo.tensor<[4], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.pad.repeat([[VARG0]], [[V0]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[32, 73], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[32, 73], f32>
// CHECK-NEXT: }

mo.graph @tile(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[20, 60], f32> {
    %0 = mo.constant <#M.dense_array<10, 20> : tensor<2xsi32>> : !mo.tensor<[2], si32>
    %1 = mo.tile(%arg0, %0) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[20, 60], f32>
    mo.output %1 : !mo.tensor<[20, 60], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @tile([[VARG0:%.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[20, 60], f32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<10, 20> : tensor<2xsi32>> : !mo.tensor<[2], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.tile([[VARG0]], [[V0]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[20, 60], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[20, 60], f32>
// CHECK-NEXT: }

mo.graph @partially_parameterized_signature<in, D6, D7>(%arg0: !mo.tensor<[in, D6, D7], f32>) -> !mo.tensor<[in, 1, D7], f32> {
    %1 = mo.reduce.mul(%arg0) [axis = 1] : (!mo.tensor<[in, D6, D7], f32>) -> !mo.tensor<[in, 1, D7], f32>
    mo.output %1 : !mo.tensor<[in, 1, D7], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @partially_parameterized_signature<in, [[D6:.*]], [[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[in, [[D6]], [[D7]]], f32>) -> !mo.tensor<[in, 1, [[D7]]], f32> {
// CHECK-NEXT:   [[V1:%.*]] = mo.reduce.mul([[VARG0]]) [axis = 1] : (!mo.tensor<[in, [[D6]], [[D7]]], f32>) -> !mo.tensor<[in, 1, [[D7]]], f32>
// CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[in, 1, [[D7]]], f32>
// CHECK-NEXT: }

mo.graph @binary_logical_op<D7, D8, D9, D10>(%arg0: !mo.tensor<[D7, D8], bool>, %arg1: !mo.tensor<[D9, D10], bool>) -> !mo.tensor<[D7, D8], bool> {
    %0 = mo.rebind(%arg1) : !mo.tensor<[D9, D10], bool> -> !mo.tensor<[D7, D8], bool>
    kgen.param.declare SORS_CH18: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[and] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> D10, D8), identical(:scalar<si64> D7, D9)))>
    %1 = mo.and(%arg0, %0) : !mo.tensor<[D7, D8], bool>
    kgen.param.declare SORS_CH6: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[xor] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> D10, D8), identical(:scalar<si64> D7, D9)))>
    %2 = mo.xor(%1, %0) : !mo.tensor<[D7, D8], bool>
    mo.output %2 : !mo.tensor<[D7, D8], bool>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @binary_logical_op<[[D7:.*]], [[D8:.*]], [[D9:.*]], [[D10:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D7]], [[D8]]], bool>, [[VARG1:%.*]]: !mo.tensor<[[[D9]], [[D10]]], bool>) -> !mo.tensor<[[[D7]], [[D8]]], bool> {
// CHECK-NEXT:   [[V0:%.*]] = mo.rebind([[VARG1]]) : !mo.tensor<[[[D9]], [[D10]]], bool> -> !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   kgen.param.declare SORS_CH18: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[and] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> [[D10]], [[D8]]), identical(:scalar<si64> [[D7]], [[D9]])))>
// CHECK-NEXT:   [[V1:%.*]] = mo.and([[VARG0]], [[V0]]) : !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   kgen.param.declare SORS_CH6: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[xor] input shapes must match" : !kgen.string}, and(identical(:scalar<si64> [[D10]], [[D8]]), identical(:scalar<si64> [[D7]], [[D9]])))>
// CHECK-NEXT:   [[V2:%.*]] = mo.xor([[V1]], [[V0]]) : !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT:   mo.output [[V2]] : !mo.tensor<[[[D7]], [[D8]]], bool>
// CHECK-NEXT: }

mo.graph @top_k(%arg0: !mo.tensor<[5, 5], si64>, %arg1: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
    %0 = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>> : !mo.tensor<[5, 5], si64>
    %1 = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>
    %2 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
    %values, %indices = mo.top_k(%0, %1, %2, %arg1) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
    mo.output %values, %indices : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @top_k([[VARG0:%.*]]: !mo.tensor<[5, 5], si64>, [[VARG1:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>> : !mo.tensor<[5, 5], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT:   [[VVALUES:%.*]], [[VINDICES:%.*]] = mo.top_k([[V0]], [[V1]], [[V2]], [[VARG1]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
// CHECK-NEXT:   mo.output [[VVALUES]], [[VINDICES]] : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
// CHECK-NEXT: }

// Ensure dynamic K properly generates an assert
// dummy is just meant to help define the param 'A'
mo.graph @top_k_dynamic<A>(%arg0: !mo.tensor<[5, 5], si64>, %arg1: !mo.tensor<[], bool>, %arg2: !mo.tensor<[A], bool>) -> (!mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>) {
    %0 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
    %1 = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>> : !mo.tensor<[5, 5], si64>
    %2 = mosh.param.to_value = <A>
    %3 = mo.index.to_tensor(%2) -> !mo.tensor<[], si64>
    kgen.param.declare assert8: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[top/bottom-k] k must be between [0, dim_size]." : !kgen.string}, lt(:scalar<si64> A, 6))>
    %values, %indices = mo.top_k(%1, %3, %0, %arg1) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>)
    mo.output %values, %indices : !mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @top_k_dynamic<A>([[VARG0:[^ ]*]]: !mo.tensor<[5, 5], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[], bool>, [[VARG2:[^ ]*]]: !mo.tensor<[A], bool>) -> (!mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>) {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>> : !mo.tensor<[5, 5], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mosh.param.to_value = <A>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.index.to_tensor([[V2]]) -> !mo.tensor<[], si64>
// CHECK-NEXT: kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[top/bottom-k] k must be between [0, dim_size]." : !kgen.string}, lt(:scalar<si64> A, 6))>
// CHECK-NEXT: [[VVALUES:[^ ]*]], [[VINDICES:[^ ]*]] = mo.top_k([[V1]], [[V3]], [[V0]], [[VARG1]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>)
// CHECK-NEXT: mo.output [[VVALUES]], [[VINDICES]] : !mo.tensor<[5, A], si64>, !mo.tensor<[5, A], si64>

mo.graph @bottom_k(%arg0: !mo.tensor<[5, 5], si64>, %arg1: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
    %0 = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>> : !mo.tensor<[5, 5], si64>
    %1 = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>
    %2 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
    %values, %indices = mo.bottom_k(%0, %1, %2, %arg1) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
    mo.output %values, %indices : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bottom_k([[VARG0:%.*]]: !mo.tensor<[5, 5], si64>, [[VARG1:%.*]]: !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>) {
// CHECK-NEXT:   [[V0:%.*]] = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24> : tensor<5x5xsi64>> : !mo.tensor<[5, 5], si64>
// CHECK-NEXT:   [[V1:%.*]] = mo.constant <#M.dense_array<3> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT:   [[V2:%.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
// CHECK-NEXT:   [[VVALUES:%.*]], [[VINDICES:%.*]] = mo.bottom_k([[V0]], [[V1]], [[V2]], [[VARG1]]) : (!mo.tensor<[5, 5], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>) -> (!mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>)
// CHECK-NEXT:   mo.output [[VVALUES]], [[VINDICES]] : !mo.tensor<[5, 3], si64>, !mo.tensor<[5, 3], si64>
// CHECK-NEXT: }

mo.graph @unsqueeze_shape_dynamic_shape<D1, D2 -> D3>(%arg0: !mo.tensor<[D1], si32>, %arg1: !mo.tensor<[D2], si32>) -> !mo.tensor<[D3], si32> {
    %0 = mo.unsqueeze_shape(%arg0, %arg1) : (!mo.tensor<[D1], si32>, !mo.tensor<[D2], si32>) -> !mo.tensor<[add(D1, D2)], si32>
    mo.output<add(D1, D2)> %0 : !mo.tensor<[add(D1, D2)], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @unsqueeze_shape_dynamic_shape<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32> {
// CHECK-NEXT:   kgen.param.declare [[EXPR:.*]]: scalar<si64> = <add([[D1]], [[D2]])>
// CHECK-NEXT:   [[V0:%.*]] = mo.unsqueeze_shape([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[EXPR]]], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[V0]]) : !mo.tensor<[[[EXPR]]], si32> -> !mo.tensor<[add([[D1]], [[D2]])], si32>
// CHECK-NEXT:   mo.output<add([[D1]], [[D2]])> [[V1]] : !mo.tensor<[add([[D1]], [[D2]])], si32>
// CHECK-NEXT: }

mo.graph @broadcast_shape_dynamic<D1, D2 -> D3>(%arg0: !mo.tensor<[D1], si32>, %arg1: !mo.tensor<[D2], si32>) -> !mo.tensor<[D3], si32> {
    %0 = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[D1], si32>, !mo.tensor<[D2], si32>) -> !mo.tensor<[max(D1, D2)], si32>
    mo.output<max(D1, D2)> %0 : !mo.tensor<[max(D1, D2)], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_dynamic<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>(%[[ARG0:.*]]: !mo.tensor<[[[D1]]], si32>, %[[ARG1:.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32>
// CHECK-NEXT: kgen.param.declare [[EXPR:.*]]: scalar<si64> = <max([[D1]], [[D2]])>
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_shape(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[EXPR]]], si32>
// CHECK-NEXT: %[[V1:.*]] = mo.rebind(%[[V0]]) : !mo.tensor<[[[EXPR]]], si32> -> !mo.tensor<[max([[D1]], [[D2]])], si32>
// CHECK-NEXT: mo.output<{{.*}}> %[[V1]]

mo.graph @broadcast_shape_static(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[5], si32>) -> !mo.tensor<[5], si32> {
    %0 = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[5], si32>) -> !mo.tensor<[5], si32>
    mo.output %0 : !mo.tensor<[5], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_static(%[[ARG0:.*]]: !mo.tensor<[3], si32>, %[[ARG1:.*]]: !mo.tensor<[5], si32>) -> !mo.tensor<[5], si32>
// CHECK-NEXT: %[[V0:.*]] = mo.broadcast_shape(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3], si32>, !mo.tensor<[5], si32>) -> !mo.tensor<[5], si32>
// CHECK-NEXT: mo.output %[[V0]] : !mo.tensor<[5], si32>

mo.graph @gather_nd_with_last_dim<D1, D2, D3, D4>(%arg0: !mo.tensor<[D1, D2, D3], si64>, %arg1: !mo.tensor<[D4, 1], si64>) -> !mo.tensor<[D1, D3], si64> {
    %0 = mo.gather_nd(%arg0, %arg1) [batchDims = 1] : (!mo.tensor<[D1, D2, D3], si64>, !mo.tensor<[D4, 1], si64>) -> !mo.tensor<[D1, D3], si64>
    mo.output %0 : !mo.tensor<[D1, D3], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @gather_nd_with_last_dim<[[D1:.*]], [[D2:.*]], [[D3:.*]], [[D4:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, [[VARG1:%.*]]: !mo.tensor<[[[D4]], 1], si64>)
// CHECK-NEXT:   [[V0:%.*]] = mo.gather_nd([[VARG0]], [[VARG1]]) [batchDims = 1] : (!mo.tensor<[[[D1]], [[D2]], [[D3]]], si64>, !mo.tensor<[[[D4]], 1], si64>) -> !mo.tensor<[[[D1]], [[D3]]], si64>
// CHECK-NEXT:   mo.output [[V0]]
// CHECK-NEXT: }

mo.graph @squeeze_shape_dynamic_shape<D1, D2 -> D3>(%arg0: !mo.tensor<[D1], si32>, %arg1: !mo.tensor<[D2], si32>) -> !mo.tensor<[D3], si32> {
    kgen.param.declare assert8: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[squeeze_shape] cannot remove more dimensions than there exists" : !kgen.string}, le(:scalar<si64> D2, D1))>
    %0 = mo.squeeze_shape(%arg0, %arg1) : (!mo.tensor<[D1], si32>, !mo.tensor<[D2], si32>) -> !mo.tensor<[add(mul_no_wrap(D2, -1), D1)], si32>
    mo.output<add(mul_no_wrap(D2, -1), D1)> %0 : !mo.tensor<[add(mul_no_wrap(D2, -1), D1)], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @squeeze_shape_dynamic_shape<[[D1:.*]], [[D2:.*]] -> [[D3:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D1]]], si32>, [[VARG1:%.*]]: !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[D3]]], si32> {
// CHECK-NEXT:   kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[squeeze_shape] cannot remove more dimensions than there exists" : !kgen.string}, le(:scalar<si64> [[D2]], [[D1]]))>
// CHECK-NEXT:   kgen.param.declare [[EXPR:.*]]: scalar<si64> = <add(mul_no_wrap([[D2]], -1), [[D1]])>
// CHECK-NEXT:   [[V0:%.*]] = mo.squeeze_shape([[VARG0]], [[VARG1]]) : (!mo.tensor<[[[D1]]], si32>, !mo.tensor<[[[D2]]], si32>) -> !mo.tensor<[[[EXPR]]], si32>
// CHECK-NEXT:   [[V1:%.*]] = mo.rebind([[V0]]) : !mo.tensor<[[[EXPR]]], si32> -> !mo.tensor<[add(mul_no_wrap([[D2]], -1), [[D1]])], si32>
// CHECK-NEXT:   mo.output<{{.*}}> [[V1]]
// CHECK-NEXT: }

mo.graph @slice_entire_dynamic_axis<D6>(%arg0: !mo.tensor<[10, D6, 20], f32>) -> !mo.tensor<[10, D6, 20], f32> {
    mo.output %arg0 : !mo.tensor<[10, D6, 20], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_entire_dynamic_axis<[[D6:.*]]>([[VARG0:%.*]]: !mo.tensor<[10, [[D6]], 20], f32>) -> !mo.tensor<[10, [[D6]], 20], f32> {
// CHECK-NEXT:   mo.output [[VARG0]] : !mo.tensor<[10, [[D6]], 20], f32>
// CHECK-NEXT: }

mo.graph @slice_out_a_third_of_constant_axis<D7>(%arg0: !mo.tensor<[50, D7, 2304], f32>) -> !mo.tensor<[50, D7, 768], f32> {
    %0 = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
    %1 = mo.constant <#M.dense_array<0, 0, 768> : tensor<3xsi64>> : !mo.tensor<[3], si64>
    %2 = mosh.param.to_value: !mosh.ape = <[50, D7, 1536]>
    %3 = mo.shape.to_tensor(%2) -> !mo.tensor<[3], si64>
    %4 = mo.slice(%arg0, %1, %3, %0) : (!mo.tensor<[50, D7, 2304], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[50, D7, 768], f32>
    mo.output %4 : !mo.tensor<[50, D7, 768], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @slice_out_a_third_of_constant_axis<[[D7:.*]]>([[VARG0:%.*]]: !mo.tensor<[50, [[D7]], 2304], f32>) -> !mo.tensor<[50, [[D7]], 768], f32> {
// CHECK-DAG:    [[V0:%.*]] = mosh.param.to_value: !mosh.ape = <[50, [[D7]], 1536]>
// CHECK-DAG:    [[V1:%.*]] = mo.constant <#M.dense_array<0, 0, 768> : tensor<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-DAG:    [[V2:%.*]] = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
// CHECK-NEXT:   [[V3:%.*]] = mo.shape.to_tensor([[V0]]) -> !mo.tensor<[3], si64>
// CHECK-NEXT:   [[V4:%.*]] = mo.slice([[VARG0]], [[V1]], [[V3]], [[V2]]) : (!mo.tensor<[50, [[D7]], 2304], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[50, [[D7]], 768], f32>
// CHECK-NEXT:   mo.output [[V4]] : !mo.tensor<[50, [[D7]], 768], f32>
// CHECK-NEXT: }

mo.graph @nested_broadcast_shapes<D29, D30, D31, D32 -> D33, D34>(%arg0: !mo.tensor<[D29, D30], bool>, %arg1: !mo.tensor<[D31, D32], si32>) -> !mo.tensor<[D33, D34], si32> {
    %0 = mo.cast(%arg0) : (!mo.tensor<[D29, D30], bool>) -> !mo.tensor<[D29, D30], si32>
    kgen.param.declare assert50: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> D29, D31), identical(:scalar<si64> D29, 1), identical(:scalar<si64> D31, 1)))>
    kgen.param.declare assert58: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> D30, D32), identical(:scalar<si64> D30, 1), identical(:scalar<si64> D32, 1)))>
    kgen.param.declare assert64: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 1 (D30) must be either 1 or equal to corresponding output dimension at index 1 (cond60)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D30, 1), D32, D30), D30), identical(:scalar<si64> D30, 1)))>
    kgen.param.declare assert68: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 0 (D29) must be either 1 or equal to corresponding output dimension at index 0 (cond52)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D29, 1), D31, D29), D29), identical(:scalar<si64> D29, 1)))>
    %1 = mo.static.broadcast_to(%arg0) : !mo.tensor<[D29, D30], bool> -> !mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], bool>
    kgen.param.declare assert73: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 1 (D30) must be either 1 or equal to corresponding output dimension at index 1 (var69)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D30, 1), D32, D30), D30), identical(:scalar<si64> D30, 1)))>
    kgen.param.declare assert78: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 0 (D29) must be either 1 or equal to corresponding output dimension at index 0 (var74)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D29, 1), D31, D29), D29), identical(:scalar<si64> D29, 1)))>
    %2 = mo.static.broadcast_to(%0) : !mo.tensor<[D29, D30], si32> -> !mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], si32>
    kgen.param.declare assert83: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 1 (D32) must be either 1 or equal to corresponding output dimension at index 1 (var79)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D30, 1), D32, D30), D32), identical(:scalar<si64> D32, 1)))>
    kgen.param.declare assert88: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 0 (D31) must be either 1 or equal to corresponding output dimension at index 0 (var84)" : !kgen.string}, or(identical(:scalar<si64> cond(identical(:scalar<si64> D29, 1), D31, D29), D31), identical(:scalar<si64> D31, 1)))>
    %3 = mo.static.broadcast_to(%arg1) : !mo.tensor<[D31, D32], si32> -> !mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], si32>
    %4 = mo.select(%1, %2, %3) : (!mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], bool>, !mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], si32>, !mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], si32>) -> !mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], si32>
    mo.output<cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)> %4 : !mo.tensor<[cond(identical(:scalar<si64> D29, 1), D31, D29), cond(identical(:scalar<si64> D30, 1), D32, D30)], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @nested_broadcast_shapes<[[D29:.*]], [[D30:.*]], [[D31:.*]], [[D32:.*]] -> [[D33:.*]], [[D34:.*]]>([[VARG0:%.*]]: !mo.tensor<[[[D29]], [[D30]]], bool>, [[VARG1:%.*]]: !mo.tensor<[[[D31]], [[D32]]], si32>) -> !mo.tensor<[[[D33]], [[D34]]], si32> {
// CHECK-NEXT:   [[V0:%.*]] = mo.cast([[VARG0]]) : (!mo.tensor<[[[D29]], [[D30]]], bool>) -> !mo.tensor<[[[D29]], [[D30]]], si32>
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> [[D29]], [[D31]]), identical(:scalar<si64> [[D29]], 1), identical(:scalar<si64> [[D31]], 1)))>
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_shape] dimensions are not broadcastable" : !kgen.string}, or(identical(:scalar<si64> [[D30]], [[D32]]), identical(:scalar<si64> [[D30]], 1), identical(:scalar<si64> [[D32]], 1)))>
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 1 ([[D30]]) must be either 1 or equal to corresponding output dimension at index 1 ({{.*}})" : !kgen.string}
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 0 ([[D29]]) must be either 1 or equal to corresponding output dimension at index 0 ({{.*}})" : !kgen.string}
// CHECK-NEXT:   [[V1:%.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D29]], [[D30]]], bool> -> !mo.tensor<[[[EXPR0:.*]], [[EXPR1:.*]]], bool>
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 1 ([[D30]]) must be either 1 or equal to corresponding output dimension at index 1 ({{.*}})" : !kgen.string}
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 0 ([[D29]]) must be either 1 or equal to corresponding output dimension at index 0 ({{.*}})" : !kgen.string}
// CHECK-NEXT:   [[V2:%.*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[[[D29]], [[D30]]], si32> -> !mo.tensor<[[[EXPR0]], [[EXPR1]]], si32>
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 1 ([[D32]]) must be either 1 or equal to corresponding output dimension at index 1 ({{.*}})" : !kgen.string}
// CHECK-DAG:    kgen.param.declare {{.*}}: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[broadcast_to] input dimension at index 0 ([[D31]]) must be either 1 or equal to corresponding output dimension at index 0 ({{.*}})" : !kgen.string}
// CHECK-NEXT:   [[V3:%.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D31]], [[D32]]], si32> -> !mo.tensor<[[[EXPR0]], [[EXPR1]]], si32>
// CHECK-DAG:    kgen.param.declare [[EXPR0]]: scalar<si64> = <cond(identical(:scalar<si64> [[D29]], 1), [[D31]], [[D29]])>
// CHECK-DAG:    kgen.param.declare [[EXPR1]]: scalar<si64> = <cond(identical(:scalar<si64> [[D30]], 1), [[D32]], [[D30]])>
// CHECK-NEXT:   [[V4:%.*]] = mo.select([[V1]], [[V2]], [[V3]])
// CHECK-NEXT:   [[V5:%.*]] = mo.rebind([[V4]]) : {{.*}} -> !mo.tensor<[cond(identical(:scalar<si64> [[D29]], 1), [[D31]], [[D29]]), cond(identical(:scalar<si64> [[D30]], 1), [[D32]], [[D30]])], si32>
// CHECK-NEXT:   mo.output{{.*}} [[V5]]
// CHECK-NEXT: }

mo.graph @onnx_select_neg_ones_bcast<D9>(%arg0: !mo.tensor<[1, 32, D9, 10], si64>) -> !mo.tensor<[1, 32, D9, 10], si64> {
    mo.output %arg0 : !mo.tensor<[1, 32, D9, 10], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @onnx_select_neg_ones_bcast<[[D9:.*]]>([[VARG0:.*]]: !mo.tensor<[1, 32, [[D9]], 10], si64>) -> !mo.tensor<[1, 32, [[D9]], 10], si64> {
// CHECK-NEXT:   mo.output [[VARG0]] : !mo.tensor<[1, 32, [[D9]], 10], si64>
// CHECK-NEXT: }


// CHECK-LABEL: mo.graph @random_normal_const
// CHECK-SAME:  (
// CHECK-SAME:    %[[FLOAT:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:    %[[SEED:.*]]: !mo.tensor<[1], ui64>
// CHECK-SAME:  ) -> !mo.tensor<[128], f32> {
mo.graph @random_normal_const(%arg0: !mo.tensor<[], f32>, %arg1: !mo.tensor<[1], ui64>) -> !mo.tensor<[128], f32> {
    %0 = mo.constant <#M.dense_array<128> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %1 = mo.random.normal(%0, %arg0, %arg0, %arg1) : (!mo.tensor<[1], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[128], f32>
    mo.output %1 : !mo.tensor<[128], f32>
}

// CHECK-LABEL: mo.graph @random_normal_dyn
// CHECK-SAME:  <[[N:.*]]>(
// CHECK-SAME:    %[[DUMMY:.*]]: !mo.tensor<[[[N]]], f32>,
// CHECK-SAME:    %[[FLOAT:.*]]: !mo.tensor<[], f32>,
// CHECK-SAME:    %[[SEED:.*]]: !mo.tensor<[1], ui64>
// CHECK-SAME:  ) -> !mo.tensor<[[[N]]], f32> {
mo.graph @random_normal_dyn<N>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[], f32>, %arg2: !mo.tensor<[1], ui64>) -> !mo.tensor<[N], f32> {
    %0 = mosh.param.to_value: !mosh.ape = <[N]>
    %1 = mo.shape.to_tensor(%0) -> !mo.tensor<[1], si64>
    %2 = mo.random.normal(%1, %arg1, %arg1, %arg2) : (!mo.tensor<[1], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[1], ui64>) -> !mo.tensor<[N], f32>
    mo.output %2 : !mo.tensor<[N], f32>
}

// CHECK-LABEL: mo.graph @reshape_neg_one_dim_in_const_shape
// CHECK-SAME:   <[[D0:.*]] -> [[D5:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], 4], si32>
// CHECK-SAME:  ) -> !mo.tensor<[3, [[D5]], 2], si32> {
mo.graph @reshape_neg_one_dim_in_const_shape<D0 -> D5>(%arg0: !mo.tensor<[D0, 4], si32>) -> !mo.tensor<[3, D5, 2], si32> {
    kgen.param.declare assert6: !mo.chain = <apply(:(!kgen.scalar<bool>) -> !mo.chain "mo.assert"{message = "[reshape] input and output number of elements must match: #mosh<ape[D0, 4]> : !mosh.ape != #mosh<ape[3, var_div2, 2]> : !mosh.ape\0AIf you are confident this is correct, consider a rebind to assert that the reshape is valid.\0AEx: ```\0A    n, m = x.shape\0A    x = x.rebind([(n // 4) * 4, m])\0A    x.reshape([n // 4, 4, m])\0A```" : !kgen.string}, identical(:scalar<si64> mul_no_wrap(div(mul_no_wrap(D0, 2), 3), 6), mul_no_wrap(D0, 4)))>
    %0 = mo.static.reshape(%arg0) : !mo.tensor<[D0, 4], si32> -> !mo.tensor<[3, div(mul_no_wrap(D0, 2), 3), 2], si32>
    mo.output<div(mul_no_wrap(D0, 2), 3)> %0 : !mo.tensor<[3, div(mul_no_wrap(D0, 2), 3), 2], si32>
}

// CHECK-LABEL: mo.graph @reshape_flatten_const_shape
// CHECK-SAME:   <[[D0:.*]] -> [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], 4], si32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D2]]], si32> {
mo.graph @reshape_flatten_const_shape<D0 -> D2>(%arg0: !mo.tensor<[D0, 4], si32>) -> !mo.tensor<[D2], si32> {
    %0 = mo.static.reshape(%arg0) : !mo.tensor<[D0, 4], si32> -> !mo.tensor<[mul_no_wrap(D0, 4)], si32>
    mo.output<mul_no_wrap(D0, 4)> %0 : !mo.tensor<[mul_no_wrap(D0, 4)], si32>
}

// CHECK-LABEL: mo.graph @e2e_fold
// CHECK-SAME:  <[[M:.*]], [[N:.*]]>(
// CHECK-SAME:   %[[ARG0:.*]]: !mo.tensor<[[[M]], 1], f32>,
// CHECK-SAME:   %[[ARG1:.*]]: !mo.tensor<[1, [[N]]], f32>
// CHECK-SAME:   %[[ARG2:.*]]: !mo.tensor<[[[N]], [[M]]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[N]], [[M]]], f32> {
mo.graph @e2e_fold<M, N>(%arg0: !mo.tensor<[M, 1], f32>, %arg1: !mo.tensor<[1, N], f32>, %arg2: !mo.tensor<[N, M], f32>) -> !mo.tensor<[N, M], f32> {
    %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[M, 1], f32> -> !mo.tensor<[M, N], f32>
    %1 = mo.static.reshape(%0) : !mo.tensor<[M, N], f32> -> !mo.tensor<[N, M], f32>
    mo.output %1 : !mo.tensor<[N, M], f32>
}

// CHECK-LABEL: mo.graph @consecutive_rebind_during_staticization
// CHECK-SAME:   %[[ARG1:.*]]: !mo.tensor
mo.graph @consecutive_rebind_during_staticization<D21, D22>(%arg0: !mo.tensor<[D21, D22], si32>) -> !mo.tensor<[D21, D22], si32> {
    mo.output %arg0 : !mo.tensor<[D21, D22], si32>
}

// This is a regression test taken from mistral-7b-hf-lptg-onnx, where
// - D0 := seq_len
// - D1 := seq_len + past_seq_len
// - D2 := batch_size
//
// Make sure everything folds to a single static broadcast.
// CHECK-LABEL: mo.graph @staticize_broadcast_on_constants
// CHECK-SAME:    <[[D0:.*]], [[D1:.*]], [[D2:.*]]>(
// CHECK-SAME:    %[[LHS:.*]]: !mo.tensor<[1, 1, [[D0]], [[D1]]], f32>,
// CHECK-SAME:    %[[RHS:.*]]: !mo.tensor<[[[D2]], 1, [[D0]], [[D1]]], f32>
//
// CHECK-NEXT:    %[[RES:.*]] = mo.static.broadcast_to(%[[LHS]]) : {{.*}} -> !mo.tensor<[[[D2]], 1, [[D0]], [[D1]]], f32>
// CHECK-NEXT:    mo.output %[[RES]]
mo.graph @staticize_broadcast_on_constants<D0, D1, D2>(%arg0: !mo.tensor<[1, 1, D0, D1], f32>, %arg1: !mo.tensor<[D2, 1, D0, D1], f32>) -> !mo.tensor<[D2, 1, D0, D1], f32> {
    %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[1, 1, D0, D1], f32> -> !mo.tensor<[D2, 1, D0, D1], f32>
    mo.output %0 : !mo.tensor<[D2, 1, D0, D1], f32>
}

// CHECK-LABEL: mo.graph @reshape_infer_dim_folded
//  CHECK-SAME:  <[[D0:.*]], [[D1:.*]]>(
//  CHECK-SAME:  %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
mo.graph @reshape_infer_dim_folded<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D1, D0], f32> {
    %0 = mo.static.reshape(%arg0) : !mo.tensor<[D0, D1], f32> -> !mo.tensor<[D1, D0], f32>
    mo.output %0 : !mo.tensor<[D1, D0], f32>
}

// This is created from `replitv15-3B-hf-context-encoding-bf16-onnx`, as a
// regression test to make sure we can staticize `mo.negative`.
// CHECK-LABEL: mo.graph @staticize_mo_negative
//  CHECK-SAME:  <[[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
//  CHECK-SAME:  %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], bf16>
//       CHECK:  %[[RES:.*]] = mo.range{{.*}} -> !mo.tensor<[[[D2]]], si64>
//  CHECK-NEXT:  mo.output{{.*}} %[[RES:.*]]
mo.graph @staticize_mo_negative<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1, D2, D3], bf16>) -> !mo.tensor<[D2], si64> {
    %0 = mo.constant <#M.dense_array<0> : !M.array<1xsi64>> : !mo.tensor<[], si64>
    %1 = mo.constant <#M.dense_array<-1> : tensor<1xsi64>> : !mo.tensor<[], si64>
    %2 = mosh.param.to_value = <mul_no_wrap(D2, -1)>
    %3 = mo.index.to_tensor(%2) -> !mo.tensor<[], si64>
    %4 = mo.range(%0, %3, %1) : (!mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D2], si64>
    mo.output %4 : !mo.tensor<[D2], si64>
}

// This checks we can use range-based folding to simplify patterns like `lt(X +
// Y, X)`. We see such pattern generated from `mo.slice` shape function in
// `replitv15-3B-hf-context-encoding-bf16-onnx`.
// CHECK-LABEL: mo.graph @fold_non_trivial_lt
//  CHECK-SAME:  <[[X:.*]], [[Y:.*]]>(
//  CHECK-SAME:  %[[INPUT:.*]]: !mo.tensor<[[[X]], [[Y]]], f32>
//  CHECK-NEXT:   mo.output %[[INPUT]]
mo.graph @fold_non_trivial_lt<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) -> !mo.tensor<[X, Y], f32> {
    mo.output %arg0 : !mo.tensor<[X, Y], f32>
}

// Test that reshape correctly handles zero dimensions in both input and output shapes.
// CHECK-LABEL: mo.graph @reshape_parametric_shape_inferred_neg_one_empty
// CHECK-SAME:   <[[D0:.*]] -> [[D2:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], 0], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[[D2]], 4, 0], f32> {
mo.graph @reshape_parametric_shape_inferred_neg_one_empty<D0 -> D2>(%arg0: !mo.tensor<[D0, 0], f32>) -> !mo.tensor<[D2, 4, 0], f32> {
    %0 = mo.static.reshape(%arg0) : !mo.tensor<[D0, 0], f32> -> !mo.tensor<[div(D0, 4), 4, 0], f32>
    mo.output<div(D0, 4)> %0 : !mo.tensor<[div(D0, 4), 4, 0], f32>
}

// CHECK-LABEL: mo.graph @split_shape
//  CHECK-SAME:  <[[N:.*]]>(
//  CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[[[N]], 20], f32>
mo.graph @split_shape<N>(%arg0: !mo.tensor<[N, 20], f32>) -> (!mo.tensor<[N, 5], f32>, !mo.tensor<[N, 15], f32>) {
    %0 = mo.constant <#M.dense_array<5, 15> : tensor<2xsi64>> : !mo.tensor<[2], si64>
    %2:2 = mo.split(%arg0, %0) [axis = -1] : (!mo.tensor<[N, 20], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[N, 5], f32>, !mo.tensor<[N, 15], f32>)
    mo.output %2#0, %2#1 : !mo.tensor<[N, 5], f32>, !mo.tensor<[N, 15], f32>
}

// CHECK-LABEL: mo.graph @roi_align
//  CHECK-SAME:   <[[C:.*]], [[M:.*]]>(
//  CHECK-SAME:    %[[INPUT:.*]]: !mo.tensor<[1, 2, 3, [[C]]], f32>,
//  CHECK-SAME:    %[[ROIS:.*]]: !mo.tensor<[[[M]], 5], f32>
mo.graph @roi_align<C, M>(%arg0: !mo.tensor<[1, 2, 3, C], f32>, %arg1: !mo.tensor<[M, 5], f32>) -> !mo.tensor<[M, 10, 20, C], f32> {
    %0 = mo.constant <#M.dense_array<10> : tensor<si64>> : !mo.tensor<[], si64>
    %1 = mo.constant <#M.dense_array<20> : tensor<si64>> : !mo.tensor<[], si64>
    %2 = mo.constant <#M.dense_array<1.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
    %3 = mo.constant <#M.dense_array<2.000000e+00> : tensor<1xf32>> : !mo.tensor<[], f32>
    %4 = mo.roi_align(%arg0, %arg1, %0, %1, %2, %3) [aligned = false, mode = "AVG"] : (!mo.tensor<[1, 2, 3, C], f32>, !mo.tensor<[M, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[M, 10, 20, C], f32>
    mo.output %4 : !mo.tensor<[M, 10, 20, C], f32>
}

// NOTE this pattern shows up from RMO lowering
// CHECK-LABEL: mo.graph @broadcast_shape_with_rebound_input
//  CHECK-SAME: <[[D0:.*]]>([[INPUT:%.*]]: !mo.tensor<[10, [[D0]], 1], f32>
mo.graph @broadcast_shape_with_rebound_input<D0>(%arg0: !mo.tensor<[10, D0, 1], f32>) -> !mo.tensor<[10, D0, 2], f32> {
    %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[10, D0, 1], f32> -> !mo.tensor<[10, D0, 2], f32>
    mo.output %0 : !mo.tensor<[10, D0, 2], f32>
}
