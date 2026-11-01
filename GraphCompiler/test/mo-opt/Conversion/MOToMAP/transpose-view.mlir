// RUN: mo-opt %s --mogg-import-packages --mo-to-map | FileCheck %s

// Verify that mo.transpose lowers to a standalone (unfused) map.iter.elementwise
// over the output shape, with a map.index.transpose op computing the input
// index -- not map.iter.opaque. The perm rides in as an operand (like slice's
// bounds): a mo.constant perm is also baked into the spec as literals, and a
// graph-argument perm shows up as `?` (ShapedType::kDynamic) there instead.

// 2D swap: out.dim[0] = in.dim[1], out.dim[1] = in.dim[0].

// CHECK-LABEL: mo.graph @transposeSwap
// CHECK-NOT:     map.iter.opaque
// mo.transpose is sync-launch, so the foreach runs immediately.
// CHECK:         [[OUT:%.*]] = map.iter.elementwise<[3, 2]> dispatch(immediate)
// CHECK:         inputs  ([[INPUT:%.*]]: !mo.tensor<[2, 3], f32>, [[PERM:%.*]]: !mo.tensor<[2], si64>)
// CHECK:         outputs ([[OUTARG:%.*]]: !mo.tensor<[3, 2], f32>)
// CHECK:         index   ([[IDX:%.*]]: !map.index<2>) -> !mo.tensor<[3, 2], f32> {
// CHECK:         [[IDX_:%.*]] = map.index.transpose<#map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>> ([[IDX]], [[PERM]])
// CHECK:         [[VAL:%.*]] = map.load [[INPUT]]{{\[}}[[IDX_]]] : !mo.tensor<[2, 3], f32>
// CHECK:         map.store [[OUTARG]]{{\[}}[[IDX]]] = [[VAL]] : !mo.tensor<[3, 2], f32>
mo.graph @transposeSwap(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// 3D permutation: out.dim[i] = in.dim[perm[i]] for perm = [2, 0, 1].

// CHECK-LABEL: mo.graph @transpose3D
// CHECK-NOT:     map.iter.opaque
// CHECK:         [[OUT2:%.*]] = map.iter.elementwise<[4, 2, 3]>
// CHECK:         inputs  ([[INPUT2:%.*]]: !mo.tensor<[2, 3, 4], f32>, [[PERM2:%.*]]: !mo.tensor<[3], si64>)
// CHECK:         outputs ([[OUTARG2:%.*]]: !mo.tensor<[4, 2, 3], f32>)
// CHECK:         index   ([[IDX2:%.*]]: !map.index<3>) -> !mo.tensor<[4, 2, 3], f32> {
// CHECK:         [[IDX2_:%.*]] = map.index.transpose<#map.index.transpose.spec<[2, 3, 4] -> [4, 2, 3], perm = [2, 0, 1]>> ([[IDX2]], [[PERM2]])
// CHECK:         [[VAL2:%.*]] = map.load [[INPUT2]]{{\[}}[[IDX2_]]] : !mo.tensor<[2, 3, 4], f32>
// CHECK:         map.store [[OUTARG2]]{{\[}}[[IDX2]]] = [[VAL2]] : !mo.tensor<[4, 2, 3], f32>
mo.graph @transpose3D(%input: !mo.tensor<[2, 3, 4], f32>) -> !mo.tensor<[4, 2, 3], f32> {
  %perm = mo.constant <#M.dense_array<2, 0, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[4, 2, 3], f32>
  mo.output %res : !mo.tensor<[4, 2, 3], f32>
}

// Symbolic dim (N): proving the lowering works when the static shape isn't
// fully concrete either.

// CHECK-LABEL: mo.graph @transposeSymbolicDim
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise<[64, N]>
// CHECK:         map.index.transpose<#map.index.transpose.spec<[N, 64] -> [64, N], perm = [1, 0]>>
mo.graph @transposeSymbolicDim<N>(%input: !mo.tensor<[N, 64], f32>) -> !mo.tensor<[64, N], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[N, 64], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[64, N], f32>
  mo.output %res : !mo.tensor<[64, N], f32>
}

// Dynamic perm: the permutation comes from a graph argument, not a
// mo.constant, so shape inference gives fully symbolic output dims and the
// spec's perm is all `?` -- the runtime perm operand supplies it. This is the
// eager interpreter's one-graph-per-op transpose. It still lowers to a view
// (map.index.transpose over the output shape), never to map.iter.opaque.

// CHECK-LABEL: mo.graph @transposeDynamicPerm
// CHECK-NOT:     map.iter.opaque
// CHECK:         map.iter.elementwise
// CHECK:         [[IDXD_:%.*]] = map.index.transpose<#map.index.transpose.spec<[D0, D1, D2, D3] -> [{{.*}}], perm = [?, ?, ?, ?]>> ({{.*}})
mo.graph @transposeDynamicPerm<D0, D1, D2, D3 -> D4, D5, D6, D7>(
    %input: !mo.tensor<[D0, D1, D2, D3], f32>,
    %perm: !mo.tensor<[4], si32>)
    -> !mo.tensor<[D4, D5, D6, D7], f32> {
  %res = mo.transpose<() -> D8, D9, D10, D11>(%input, %perm) : (
    !mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[4], si32>
  ) -> !mo.tensor<[D8, D9, D10, D11], f32>
  mo.output<D8, D9, D10, D11> %res : !mo.tensor<[D8, D9, D10, D11], f32>
}
