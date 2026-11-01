// RUN: mo-opt %s --mogg-anonymize-dim-names | FileCheck %s
// RUN: mo-opt %s --mogg-anonymize-dim-names --mogg-anonymize-dim-names | FileCheck %s

// A symbolic dim name on a stub's boundary would block MergeIdenticalFuncs
// from recognizing two structurally-identical stubs that differ only in a dim's
// name as duplicates, so this pass strips it -- from the block arg's type, the
// function-type signature, and any in-body use (changing a block arg's type
// updates its uses automatically). Idempotent: the second RUN line reapplies
// the pass a second time and checks the output is unchanged.

// CHECK-LABEL: mogg.func @named_dim private
// CHECK-SAME: (%[[ARG:.*]]: !mogg.dim)
// CHECK: mogg.shape.create(%[[ARG]]) {{.*}} : !mogg.dim
// CHECK-NOT: !mogg.dim<
mogg.func @named_dim private dispatch(deferred)(%d: !mogg.dim<batch>)
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"named_dim">>} {
  %s = mogg.shape.create(%d) <shape = #mosh<ape[batch, 4]> : !mosh.ape> : !mogg.dim<batch>
  mogg.output
}

// A stub with no dim args at all is left alone.
// CHECK-LABEL: mogg.func @no_dims private
// CHECK-SAME: ()
mogg.func @no_dims private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"no_dims">>} {
  mogg.output
}

// An already-anonymous dim arg is left alone (no-op, not just harmless).
// CHECK-LABEL: mogg.func @already_anonymous private
// CHECK-SAME: (%{{.*}}: !mogg.dim)
mogg.func @already_anonymous private dispatch(deferred)(%d: !mogg.dim)
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"already_anonymous">>} {
  mogg.output
}
