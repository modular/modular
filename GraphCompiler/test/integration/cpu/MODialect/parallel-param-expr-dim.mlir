// RUN: mo-opt --mlir-print-local-scope %s --gc-pipeline-pre-jit | FileCheck %s

// Regression test: `mo.parallel` with parameter-expression-form dims
// (e.g. `div(D0, 2)`) in tensor types must lower cleanly through MOGGToMGP
// and into the kernel ABI without leaking a `KGEN::ParamOperatorAttr` into
// kernel scope.

!t_in = !mo.tensor<[D0, 3], f32>
!t = !mo.tensor<[div(D0, 2), 3], f32>
!b = !mo.bundle<[!t, !t]>

// CHECK-LABEL: mgp.model @parallel_param_expr_dim
mo.graph @parallel_param_expr_dim<D0>(%arg0: !t_in, %arg1: !t_in) -> (!t, !t) {
  %r0 = mo.rebind(%arg0) : !t_in -> !t
  %r1 = mo.rebind(%arg1) : !t_in -> !t
  %bundle = mo.tensor.bundle(%r0, %r1) : (!t, !t) -> (!b)
  %res_bundle = mo.parallel (%x) in (%bundle : !b) -> (!b) {
    %1 = mo.relu(%x) : !t
    mo.yield %1 : !t
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!b) -> (!t, !t)
  mo.output %res_0, %res_1 : !t, !t
}

// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = mogg.compiled.call @{{[^(]+}}({{[^)]*}}) {{(dispatch\([a-z]+\) )?}}fusionInfo = #mogg.fusion<#mogg.fusion_sequence<[{{.*}}"pop.div"{{.*}}-> index
// CHECK: mogg.compiled.call @{{[^(]+}}(%{{[a-zA-Z0-9_]+}}, %[[DIM]], %{{[a-zA-Z0-9_]+}}) {{(dispatch\([a-z]+\) )?}}fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mgp.buffer.alloc"
// CHECK: mogg.compiled.call @{{[^(]+}}({{[^)]*}}) {{(dispatch\([a-z]+\) )?}}fusionInfo = #mogg.fusion<#mogg.fusion_sequence<[{{.*}}"mo.relu"
