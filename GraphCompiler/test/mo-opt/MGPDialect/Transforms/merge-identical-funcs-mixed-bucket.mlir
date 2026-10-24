// RUN: mo-opt %s -mogg-merge-identical-funcs | FileCheck %s

// Regression test: a kernel-hash bucket can hold both `mogg.call`
// entries (added when an in-graph kernel-to-kernel callsite is dedup'd
// at MergeIdenticalFuncs.cpp:204) and `StringAttr`-only entries
// with `callOp == nullptr` (added when a top-level FuncOp is
// dedup'd via the `node->callers.empty()` path at line 236).  Earlier
// the post-processing rewrite loop only checked `finalCandidate.callOp`
// and then unconditionally dereferenced `callOp.callOp` for every entry,
// segfaulting whenever the chosen `finalCandidate` was a `CallOp`
// and the bucket also held a null entry.
//
// Trigger mechanics, level-by-level:
//   * `@inner` is a leaf with `@caller` in `node->callers`, so the
//     line-225 path does NOT fire for it.  It only enters
//     `uniqueFuncSet` when `@caller` runs at level 2 and processes its
//     callsite via line 196.
//   * `@z_top` is a leaf with no in-graph callers; line 225 fires at
//     level 1 and inserts `@z_top` as the bucket's intermediate
//     canonical (because `@inner` hasn't been inserted yet).
//   * At level 2, `@caller`'s callsite finds `@z_top` and emplaces a
//     `CallOp` entry into `reverseReplacements[@z_top.symname]`.
//     The `CallStubOrCalleeName(MOGG::CallOp)` ctor captures the
//     entry's `name` field as the call.stub's *current* callee symbol,
//     which at that point is still `@inner` (the rewrite at line 215
//     has not run yet).  So the bucket ends up holding
//     `[StringAttr("z_top"), CallOp(name="inner")]` -- two distinct
//     names.
//   * Post-processing's `min_element` picks `"inner"` as
//     `finalCandidate`, a `CallOp`.  The buggy loop then iterates
//     `replacedCallOps` and tries to `setName` the StringAttr entry,
//     dereferencing a null `callOp.callOp` and crashing.
//   * The fix null-guards each iteration; the StringAttr entry is
//     skipped, and only the `CallOp` gets `setName("inner")`.

// Both kernel definitions still survive -- this pass only rewrites
// references; symbol-DCE would prune duplicates later.
// CHECK-DAG: mogg.func @inner private
// CHECK-DAG: mogg.func @z_top private
mogg.func @inner private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"inner">>} {
  mogg.output
}

mogg.func @z_top private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"z_top">>} {
  mogg.output
}

// `@caller`'s callsite gets renamed to point at the bucket's
// alphabetically smallest entry name, which (per the trace above) is
// `@inner` -- the original callee symbol captured at emplace time.
// CHECK-LABEL: @caller
// CHECK:      mogg.call @inner
// CHECK-NEXT: mogg.output
mogg.func @caller private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"caller">>} {
  mogg.call @inner
  mogg.output
}
