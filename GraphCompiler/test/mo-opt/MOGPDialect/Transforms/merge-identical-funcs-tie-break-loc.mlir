// RUN: mo-opt %s -mogg-merge-identical-funcs -mlir-print-debuginfo -mlir-print-local-scope 2>&1 | FileCheck %s
//
// The -mlir-print-local-scope flag is important: without it the printer hoists
// every location into `#locN` aliases at the end of the module, and a check
// for "some fused location mentioning site_a" then matches an alias definition
// that exists whether or not the fix is in place. Inline locations tie the
// check to @a_dup's own location.

// Regression test for the profile_scope-motivated location-loss fix: the
// hash-set insertion winner (the stub that accumulates every merged
// duplicate's location, see CallGraph::doRewrite) is not always the same
// stub as the tie-break winner (the lexicographically smallest name, chosen
// in mergeIdenticalKernelStubs for deterministic output). When they differ,
// the surviving symbol must still carry the accumulated location, or a
// kernel body genuinely shared by two profile scopes would silently
// report as belonging to only one of them.
//
// Trigger mechanics (mirrors merge-identical-funcs-mixed-bucket.mlir):
//   * @z_dup is a leaf with no in-graph callers, so it enters uniqueFuncSet
//     directly at level 1 (CallGraph::doRewrite's `node->callers.empty()`
//     path) and becomes the insertion winner: the accumulator that ends up
//     with a FusedLoc of both stubs' locations once @a_dup merges into it.
//   * @a_dup is called by @caller, so it only enters uniqueFuncSet when
//     @caller's callsite loop runs at level 2 -- strictly after @z_dup's
//     level-1 insertion has already completed.
//   * @a_dup's name sorts before @z_dup's, so the stabilization step's
//     min_element picks @a_dup as the tie-break winner: every call site is
//     rewritten to @a_dup, even though @z_dup was the one holding the
//     accumulated location.

// @a_dup (the surviving symbol) carries a FusedLoc combining both stubs'
// original source notes -- proving the accumulated location moved over rather
// than being silently dropped. Without the fix @a_dup keeps only its own
// `fused<...site_a...>` location and there is no enclosing `fused[`.
// CHECK-LABEL: mogg.func @a_dup private
// CHECK: loc(fused[fused<{{.*}}site_a{{.*}}>[{{.*}}], fused<{{.*}}site_z{{.*}}>[{{.*}}]])

// The call site is rewritten to the tie-break winner, not the insertion
// winner.
// CHECK-LABEL: @caller
// CHECK: mogg.call @a_dup

mogg.func @z_dup private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"z_dup">>} {
  mogg.output
} loc(fused<"  File \"model.py\", line 10, in site_z\n">["model.py":10:0])

mogg.func @a_dup private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"a_dup">>} {
  mogg.output
} loc(fused<"  File \"model.py\", line 20, in site_a\n">["model.py":20:0])

mogg.func @caller private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"caller">>} {
  mogg.call @a_dup
  mogg.output
}
