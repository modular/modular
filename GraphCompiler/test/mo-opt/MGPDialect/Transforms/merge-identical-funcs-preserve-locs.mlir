// RUN: mo-opt %s -mogg-merge-identical-funcs -mlir-print-debuginfo 2>&1 | FileCheck %s

// Test that MergeIdenticalFuncs preserves source note locations from
// both stubs when merging duplicates. Without location merging, the losing
// stub's Python call site metadata would be silently dropped.

// func_a and func_b are structurally identical (same body, same attributes)
// but carry different source note locations. After merging, the surviving
// stub should have a FusedLoc combining both original source notes.

// Verify call sites are rewritten to the surviving stub.
// CHECK: mogg.call @func_a
// CHECK: mogg.call @func_a

// The individual source note locs survive in the #loc alias table.
// CHECK-DAG: loc(fused<{{.*}}call_site_a
// CHECK-DAG: loc(fused<{{.*}}call_site_b

// A merged fused loc references both (the bare fused[] without metadata).
// This only exists if the merge pass combined the locations.
// CHECK: loc(fused[

mogg.func @func_a private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_a">>} {
  mogg.output
} loc(fused<"  File \22model.py\22, line 10, in call_site_a\n">["model.py":10:0])

mogg.func @func_b private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"func_b">>} {
  mogg.output
} loc(fused<"  File \22model.py\22, line 20, in call_site_b\n">["model.py":20:0])

mogg.func @caller private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "cpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"caller">>} {
  mogg.call @func_a
  mogg.call @func_b
  mogg.output
}
