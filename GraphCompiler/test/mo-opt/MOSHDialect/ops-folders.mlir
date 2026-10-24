// RUN: mo-opt %s --canonicalize --cse | FileCheck %s

// CHECK: kgen.generator @concat_fold<[[SH1:.*]]: !mosh.ape, [[D1:.*]], [[D2:.*]]>(
kgen.generator @concat_fold<Sh1: !mosh.ape, D1, D2>(
// CHECK-SAME: ) -> (!mosh.ape, !mosh.ape, !mosh.ape) {
) -> (!mosh.ape, !mosh.ape, !mosh.ape) {
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]], 42]>
  // CHECK-DAG: %[[V23:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]], 42, 3, [[D2]]]>
  %v1 = mosh.param.to_value: !mosh.ape = <Sh1>
  %v2 = mosh.param.to_value: !mosh.ape = <[D1, 42]>
  %v3 = mosh.param.to_value: !mosh.ape = <[3, D2]>
  // CHECK-DAG: %[[V11:.*]] = mosh.concat(%[[V1]], %[[V1]])
  // CHECK-DAG: %[[V12:.*]] = mosh.concat(%[[V1]], %[[V2]])
  %v11 = mosh.concat(%v1, %v1)
  %v12 = mosh.concat(%v1, %v2)
  %v23 = mosh.concat(%v2, %v3)
  // CHECK-NEXT: hlcf.return %[[V11]], %[[V12]], %[[V23]]
  hlcf.return %v11, %v12, %v23 : !mosh.ape, !mosh.ape, !mosh.ape
}

// CHECK: kgen.func @concat_empty_fold() -> !mosh.ape {
kgen.func @concat_empty_fold() -> !mosh.ape {
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[]>
  %c1 = mosh.concat()
  // CHECK: hlcf.return %[[V1]]
  hlcf.return %c1 : !mosh.ape
}

// CHECK: kgen.generator @concat_unary_fold<[[SH1:.*]]: !mosh.ape>() -> (!mosh.ape, !mosh.ape)
kgen.generator @concat_unary_fold<Sh1: !mosh.ape>() -> (!mosh.ape, !mosh.ape) {
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]], 42]>
  %v1 = mosh.param.to_value: !mosh.ape = <Sh1>
  %v2 = mosh.param.to_value: !mosh.ape = <[D1, 42]>
  %c1 = mosh.concat(%v1)
  %c2 = mosh.concat(%v2)
  // CHECK: hlcf.return %[[V1]], %[[V2]]
  hlcf.return %c1, %c2 : !mosh.ape, !mosh.ape
}

// CHECK: kgen.generator @concat_variadic_fold<[[SH1:.*]]: !mosh.ape, [[D1:.*]], [[D2:.*]]>(
kgen.generator @concat_variadic_fold<Sh1: !mosh.ape, D1, D2>(
// CHECK-SAME: ) -> (!mosh.ape, !mosh.ape) {
) -> (!mosh.ape, !mosh.ape) {
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]], 42]>
  // CHECK-DAG: %[[V232:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]], 42, 3, [[D2]], [[D1]], 42]>
  %v1 = mosh.param.to_value: !mosh.ape = <Sh1>
  %v2 = mosh.param.to_value: !mosh.ape = <[D1, 42]>
  %v3 = mosh.param.to_value: !mosh.ape = <[3, D2]>
  // CHECK-DAG: %[[V121:.*]] = mosh.concat(%[[V1]], %[[V2]], %[[V1]])
  %v121 = mosh.concat(%v1, %v2, %v1)
  %v232 = mosh.concat(%v2, %v3, %v2)
  // CHECK-NEXT: hlcf.return %[[V121]], %[[V232]]
  hlcf.return %v121, %v232 : !mosh.ape, !mosh.ape
}

// CHECK: kgen.generator @concat_canonicalize<[[SH0:.*]]: !mosh.ape>(%[[SH1:.*]]: !mosh.ape, %[[SH2:.*]]: !mosh.ape) -> (!mosh.ape, !mosh.ape) {
kgen.generator @concat_canonicalize<Sh0: !mosh.ape>(%sh1: !mosh.ape, %sh2: !mosh.ape) -> (!mosh.ape, !mosh.ape) {
  // CHECK: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH0]]>
  %v1 = mosh.param.to_value: !mosh.ape = <Sh0>
  // CHECK-DAG: %[[C1:.*]] = mosh.concat(%[[V1]], %[[SH2]])
  // CHECK-DAG: %[[C2:.*]] = mosh.concat(%[[V1]], %[[SH2]], %[[SH1]], %[[V1]])
  %c1 = mosh.concat(%v1, %sh2)
  %c2 = mosh.concat(%c1, %sh1, %v1)
  // CHECK: hlcf.return %[[C1]], %[[C2]] : !mosh.ape, !mosh.ape
  hlcf.return %c1, %c2 : !mosh.ape, !mosh.ape
}

// TODO: implement tests for shapes that do not fold
//       - because of unknown shape arg
//       - because of not broadcastable shapes
//       - because it's not possible to decide without asserts if broadcastable
// CHECK: kgen.generator @broadcast_fold_static() -> (!mosh.ape, !mosh.ape) {
kgen.generator @broadcast_fold_static() -> (!mosh.ape, !mosh.ape) {
  // CHECK-NEXT: %[[RES:.*]] = mosh.param.to_value: !mosh.ape = <[5, 10, 42]>
  %s0 = mosh.param.to_value: !mosh.ape = <[5, 10, 1]>
  %s1 = mosh.param.to_value: !mosh.ape = <[10, 42]>
  %s3 = mosh.broadcast(%s0, %s1)
  %s4 = mosh.broadcast(%s1, %s0)
  // CHECK-NEXT: hlcf.return %[[RES]], %[[RES]] : !mosh.ape, !mosh.ape
  hlcf.return %s3, %s4 : !mosh.ape, !mosh.ape
}

// CHECK: kgen.generator @broadcast_fold_dyn<[[N:.*]], [[K:.*]]>() -> (!mosh.ape, !mosh.ape) {
kgen.generator @broadcast_fold_dyn<N, K>() -> (!mosh.ape, !mosh.ape) {
  // CHECK: %[[RES:.*]] = mosh.param.to_value: !mosh.ape = <[[[N]], 10, 42]>
  %s0 = mosh.param.to_value: !mosh.ape = <[N, 10, 1]>
  %s1 = mosh.param.to_value: !mosh.ape = <[10, 42]>
  %s2 = mosh.param.to_value: !mosh.ape = <[1, 10, 42]>
  %s3 = mosh.broadcast(%s0, %s1)
  %s4 = mosh.broadcast(%s0, %s2)
  // CHECK-NEXT: hlcf.return %[[RES]], %[[RES]] : !mosh.ape, !mosh.ape
  hlcf.return %s3, %s4 : !mosh.ape, !mosh.ape
}

// CHECK: kgen.generator @broadcast_not_fold<[[N:.*]], [[K:.*]], [[SH:.*]]: !mosh.ape>()
kgen.generator @broadcast_not_fold<N, K, Sh: !mosh.ape>()
    // CHECK-SAME: -> (!mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape) {
    -> (!mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape) {
  // CHECK-DAG: %[[VAL0:.*]] = mosh.param.to_value: !mosh.ape = <[[[N]], 10, 42]>
  // CHECK-DAG: %[[VAL1:.*]] = mosh.param.to_value: !mosh.ape = <[[[N]], 10, 5]>
  // CHECK-DAG: %[[VAL2:.*]] = mosh.param.to_value: !mosh.ape = <[[[K]], 10, 42]>
  // CHECK-DAG: %[[VAL3:.*]] = mosh.param.to_value: !mosh.ape = <[1, 10, [[K]]]>
  // CHECK-DAG: %[[VAL4:.*]] = mosh.param.to_value: !mosh.ape = <[[SH]]>
  // CHECK-DAG: %[[VAL5:.*]] = mosh.broadcast(%[[VAL0]], %[[VAL1]])
  // CHECK-DAG: %[[VAL6:.*]] = mosh.broadcast(%[[VAL0]], %[[VAL2]])
  // CHECK-DAG: %[[VAL7:.*]] = mosh.broadcast(%[[VAL0]], %[[VAL3]])
  // CHECK-DAG: %[[VAL8:.*]] = mosh.broadcast(%[[VAL0]], %[[VAL4]])
  %s0 = mosh.param.to_value: !mosh.ape = <[N, 10, 42]>
  %s1 = mosh.param.to_value: !mosh.ape = <[N, 10, 5]>
  %s2 = mosh.param.to_value: !mosh.ape = <[K, 10, 42]>
  %s3 = mosh.param.to_value: !mosh.ape = <[1, 10, K]>
  %s4 = mosh.param.to_value: !mosh.ape = <Sh>
  %s5 = mosh.broadcast(%s0, %s1)
  %s6 = mosh.broadcast(%s0, %s2)
  %s7 = mosh.broadcast(%s0, %s3)
  %s8 = mosh.broadcast(%s0, %s4)
  // CHECK-NEXT: hlcf.return %[[VAL5]], %[[VAL6]], %[[VAL7]], %[[VAL8]] : !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape
  hlcf.return %s5, %s6, %s7, %s8 : !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape
}

// CHECK: kgen.generator @equal_fold<[[SH1:.*]]: !mosh.ape, [[D1:.*]], [[D2:.*]]>()
kgen.generator @equal_fold<Sh1: !mosh.ape, D1, D2>() -> (!kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>) {
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]], 42]>
  // CHECK-DAG: %[[TRUE:.*]] = mosh.param.to_value: scalar<bool> = <true>
  // CHECK-DAG: %[[B12:.*]] = mosh.eq(%[[V1]], %[[V2]])
  // CHECK-DAG: %[[B23:.*]] = mosh.param.to_value: scalar<bool> = <and(identical(:scalar<si64> [[D1]], 3), identical(:scalar<si64> [[D2]], 42))>
  // CHECK-DAG: %[[FALSE:.*]] = mosh.param.to_value: scalar<bool> = <false>
  // CHECK: hlcf.return %[[TRUE]], %[[B12]], %[[B23]], %[[FALSE]], %[[FALSE]] : !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>
  %v1 = mosh.param.to_value: !mosh.ape = <Sh1>
  %v2 = mosh.param.to_value: !mosh.ape = <[D1, 42]>
  %v3 = mosh.param.to_value: !mosh.ape = <[3, D2]>
  %v4 = mosh.param.to_value: !mosh.ape = <[4, D2]>
  %v5 = mosh.param.to_value: !mosh.ape = <[D1, D2, 3, 4]>
  %b11 = mosh.eq(%v1, %v1)
  %b12 = mosh.eq(%v1, %v2)
  %b23 = mosh.eq(%v2, %v3)
  %b34 = mosh.eq(%v3, %v4)
  %b25 = mosh.eq(%v2, %v5)
  hlcf.return %b11, %b12, %b23, %b34, %b25 : !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>, !kgen.scalar<bool>
}

// CHECK: kgen.generator @get_dim_fold<[[SH1:.*]]: !mosh.ape, [[D1:.*]]>()
kgen.generator @get_dim_fold<Sh1: !mosh.ape, D1>() -> (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>) {
  // CHECK-DAG: %[[C0:.*]] = mosh.param.to_value = <0>
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH1]]>
  // CHECK-DAG: %[[D10:.*]] = mosh.get_dim(%[[V1]])[%[[C0]]]
  // CHECK-DAG: %[[D20:.*]] = mosh.param.to_value = <[[D1]]>
  // CHECK-DAG: %[[D21:.*]] = mosh.param.to_value = <42>
  // CHECK-DAG: %[[D22:.*]] = mosh.param.to_value = <23>
  // CHECK: hlcf.return %[[D10]], %[[D20]], %[[D21]], %[[D22]] : !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>
  %idx0 = mosh.param.to_value = <0>
  %idx1 = mosh.param.to_value = <1>
  %idx-1 = mosh.param.to_value = <-1>
  %v1 = mosh.param.to_value: !mosh.ape = <Sh1>
  %v2 = mosh.param.to_value: !mosh.ape = <[D1, 42, 23]>
  %d10 = mosh.get_dim(%v1)[%idx0]
  %d20 = mosh.get_dim(%v2)[%idx0]
  %d21 = mosh.get_dim(%v2)[%idx1]
  %d22 = mosh.get_dim(%v2)[%idx-1]
  hlcf.return %d10, %d20, %d21, %d22 : !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>
}

// CHECK-LABEL: kgen.generator @get_dim_nofold
// CHECK: <[[SH1:.*]]: scalar<si32>>
kgen.generator @get_dim_nofold<D1: !kgen.scalar<si32>>() -> (!kgen.scalar<si64>, !kgen.scalar<si64>) {
  // CHECK-DAG: %[[C0:.*]] = mosh.param.to_value = <0>
  // CHECK-DAG: %[[C1:.*]] = mosh.param.to_value = <1>
  // CHECK-DAG: %[[S0:.*]] = mosh.param.to_value: !mosh.ape = <[:scalar<si32> 42, :scalar<si32> D1]>
  // CHECK-DAG: %[[D0:.*]] = mosh.get_dim(%[[S0]])[%[[C0]]]
  // CHECK-DAG: %[[D1:.*]] = mosh.get_dim(%[[S0]])[%[[C1]]]
  // CHECK: hlcf.return %[[D0]], %[[D1]]

  %i0 = mosh.param.to_value = <0>
  %i1 = mosh.param.to_value = <1>
  %shape = mosh.param.to_value: !mosh.ape = <[:scalar<si32> 42, :scalar<si32> D1]>

  %d0 = mosh.get_dim(%shape)[%i0]
  %d1 = mosh.get_dim(%shape)[%i1]
  hlcf.return %d0, %d1 : !kgen.scalar<si64>, !kgen.scalar<si64>
}

// CHECK: kgen.generator @get_rank_fold<[[SH1:.*]]: !mosh.ape, [[D1:.*]]>()
kgen.generator @get_rank_fold<Sh1: !mosh.ape, D1>() -> (!kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>) {
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH1]]>
  // CHECK-DAG: %[[R1:.*]] = mosh.get_rank(%[[V1]])
  // CHECK-DAG: %[[R2:.*]] = mosh.param.to_value = <2>
  // CHECK-DAG: %[[R3:.*]] = mosh.param.to_value = <0>
  // CHECK: hlcf.return %[[R1]], %[[R2]], %[[R3]] : !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>
  %v1 = mosh.param.to_value: !mosh.ape = <Sh1>
  %v2 = mosh.param.to_value: !mosh.ape = <[D1, 42]>
  %v3 = mosh.param.to_value: !mosh.ape = <[]>
  %r1 = mosh.get_rank(%v1)
  %r2 = mosh.get_rank(%v2)
  %r3 = mosh.get_rank(%v3)
  hlcf.return %r1, %r2, %r3 : !kgen.scalar<si64>, !kgen.scalar<si64>, !kgen.scalar<si64>
}

// CHECK: kgen.generator @new_fold<[[D1:.*]], [[D2:.*]]>()
kgen.generator @new_fold<D1, D2>() -> !mosh.ape {
  // CHECK: %[[S1:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]], 42, [[D2]]]>
  // CHECK: hlcf.return %[[S1]]
  %d1 = mosh.param.to_value = <D1>
  %d2 = mosh.param.to_value = <42>
  %d3 = mosh.param.to_value = <D2>
  %s1 = mosh.new(%d1, %d2, %d3)
  hlcf.return %s1 : !mosh.ape
}

// CHECK: kgen.generator @num_elements_dyn_rank_no_fold<[[SH0:.*]]: !mosh.ape>() -> !kgen.scalar<si64> {
kgen.generator @num_elements_dyn_rank_no_fold<Sh0: !mosh.ape>() -> !kgen.scalar<si64> {
  // CHECK-NEXT: %[[SH:.*]] = mosh.param.to_value: !mosh.ape = <[[SH0]]>
  %sh0 = mosh.param.to_value: !mosh.ape = <Sh0>
  // CHECK-NEXT: %[[RES:.*]] = mosh.num_elements(%[[SH]])
  %0 = mosh.num_elements(%sh0)
  // CHECK-NEXT: hlcf.return %[[RES]] : !kgen.scalar<si64>
  hlcf.return %0 : !kgen.scalar<si64>
}

// CHECK: kgen.generator @num_elements_static_rank_fold<[[D0:.*]]>() -> !kgen.scalar<si64> {
kgen.generator @num_elements_static_rank_fold<D0>() -> !kgen.scalar<si64> {
  %sh0 = mosh.param.to_value: !mosh.ape = <[42, D0, 10]>
  // CHECK-NEXT: %[[RES:.*]] = mosh.param.to_value = <mul_no_wrap([[D0]], 420)>
  %0 = mosh.num_elements(%sh0)
  // CHECK-NEXT: hlcf.return %[[RES]] : !kgen.scalar<si64>
  hlcf.return %0 : !kgen.scalar<si64>
}

// CHECK: kgen.generator @slice_fold<[[SH1:.*]]: !mosh.ape, [[D1:.*]], [[D2:.*]]>(
kgen.generator @slice_fold<Sh1: !mosh.ape, D1, D2>(
// CHECK: ) -> (!mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape) {
) -> (!mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape) {
  // CHECK-DAG: %[[C0:.*]] = mosh.param.to_value = <0>
  // CHECK-DAG: %[[CN1:.*]] = mosh.param.to_value = <-1>
  %idx0 = mosh.param.to_value = <0>
  %idx1 = mosh.param.to_value = <1>
  %idx2 = mosh.param.to_value = <2>
  %idx3 = mosh.param.to_value = <3>
  %idx99 = mosh.param.to_value = <99>
  %idx-1 = mosh.param.to_value = <-1>
  %idx-2 = mosh.param.to_value = <-2>
  %idx-99 = mosh.param.to_value = <-99>
  // CHECK-DAG: %[[V1:.*]] = mosh.param.to_value: !mosh.ape = <[[SH1]]>
  // CHECK-DAG: %[[V2:.*]] = mosh.param.to_value: !mosh.ape = <[[[D1]]]>
  // CHECK-DAG: %[[V3:.*]] = mosh.param.to_value: !mosh.ape = <[[[D2]], [[D1]]]>
  // CHECK-DAG: %[[V4:.*]] = mosh.param.to_value: !mosh.ape = <[42, [[D2]]]>
  // CHECK-DAG: %[[V5:.*]] = mosh.param.to_value: !mosh.ape = <[[[D2]]]>
  // CHECK-DAG: %[[V6:.*]] = mosh.param.to_value: !mosh.ape = <[3, [[D2]]]>
  // CHECK-DAG: %[[S1:.*]] = mosh.slice(%[[V1]])[%[[C0]], %[[CN1]]]
  // CHECK: hlcf.return %[[S1]], %[[V2]], %[[V3]], %[[V4]], %[[V5]], %[[V6]]
  %v1 = mosh.param.to_value: !mosh.ape = <Sh1>
  %v2 = mosh.param.to_value: !mosh.ape = <[D1, 42, D2]>
  %v3 = mosh.param.to_value: !mosh.ape = <[3, D2, D1]>
  %s1 = mosh.slice(%v1)[%idx0, %idx-1]
  %s2 = mosh.slice(%v2)[%idx0, %idx-2]
  %s3 = mosh.slice(%v3)[%idx1, %idx3]
  %s4 = mosh.slice(%v2)[%idx-2, %idx3]
  %s5 = mosh.slice(%v2)[%idx2, %idx99]
  %s6 = mosh.slice(%v3)[%idx-99, %idx2]
  hlcf.return %s1, %s2, %s3, %s4, %s5, %s6 : !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape, !mosh.ape
}

// CHECK: kgen.generator @param_from_value_fold<[[D0:.*]]>() -> !mosh.ape {
kgen.generator @param_from_value_fold<D0>() -> !mosh.ape {
  // CHECK: kgen.param.declare V: !mosh.ape = <[42, [[D0]], 10]>
  %sh0 = mosh.param.to_value: !mosh.ape = <[42, D0, 10]>
  mosh.param.from_value V : !mosh.ape = %sh0

  // CHECK: %[[V0:.*]] = mosh.param.to_value: !mosh.ape = <V>
  %val = mosh.param.to_value: !mosh.ape = <V>
  hlcf.return %val : !mosh.ape
}
