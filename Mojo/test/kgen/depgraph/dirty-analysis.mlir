// Exercises the dirty-propagation analysis end to end through the real file
// hasher: the test creates the source files on disk, and each `content_hash`
// below is the xxh3-64 digest (`hashModuleBuffer`) of that exact file content.
// RUN: rm -rf %t && mkdir -p %t
// RUN: printf 'content a\n' > %t/a.mojo
// RUN: printf 'content b\n' > %t/b.mojo
// RUN: printf 'content c\n' > %t/c.mojo
// RUN: sed 's|DIR|%t|g' %s > %t/graph.mlir

// Baseline: every stored hash matches its file, so only the unverifiable
// nodes (missing file, never hashed) and their dependents are dirty.
// RUN: kgen-opt -depgraph-mark-dirty %t/graph.mlir | FileCheck --check-prefix=CLEAN %s

// Surgically replace the stored hash of one node per scenario to simulate a
// changed source, then re-run the analysis.
// RUN: sed -e '/@chain_leaf/s|content_hash = "[0-9a-f]*"|content_hash = "stale"|' \
// RUN:     -e '/@dia_base/s|content_hash = "[0-9a-f]*"|content_hash = "stale"|' \
// RUN:     -e '/@cyc_a/s|content_hash = "[0-9a-f]*"|content_hash = "stale"|' \
// RUN:     %t/graph.mlir > %t/stale.mlir
// RUN: kgen-opt -depgraph-mark-dirty %t/stale.mlir | FileCheck --check-prefix=DIRTY %s

// Linear chain main -> mid -> leaf plus an independent sibling: the changed
// leaf and its transitive dependents are dirty; the sibling stays clean.
// CLEAN-LABEL: depgraph.graph root @chain_main
// CLEAN-NOT: depgraph.dirty
// DIRTY-LABEL: depgraph.graph root @chain_main
// DIRTY: depgraph.module @chain_leaf{{.*}}depgraph.dirty
// DIRTY: depgraph.module @chain_mid{{.*}}depgraph.dirty
// DIRTY: depgraph.module @chain_main{{.*}}depgraph.dirty
// DIRTY-NOT: depgraph.dirty
depgraph.graph root @chain_main {
  %0 = depgraph.module @chain_leaf <path = "DIR/a.mojo", content_hash = "807b905a30365194">
  %1 = depgraph.module @chain_mid(%0) <path = "DIR/b.mojo", content_hash = "63dd5c36f854c19e">
  %2 = depgraph.module @chain_main(%1) <path = "DIR/c.mojo", content_hash = "d9d60135c91335d8">
  %3 = depgraph.module @chain_sib <path = "DIR/a.mojo", content_hash = "807b905a30365194">
}

// Diamond: root depends on left and right, both of which depend on base.
// base changed, so every node is dirty (root reached via both paths).
// CLEAN-LABEL: depgraph.graph root @dia_root
// CLEAN-NOT: depgraph.dirty
// DIRTY-LABEL: depgraph.graph root @dia_root
// DIRTY: depgraph.module @dia_base{{.*}}depgraph.dirty
// DIRTY: depgraph.module @dia_left{{.*}}depgraph.dirty
// DIRTY: depgraph.module @dia_right{{.*}}depgraph.dirty
// DIRTY: depgraph.module @dia_root{{.*}}depgraph.dirty
depgraph.graph root @dia_root {
  %0 = depgraph.module @dia_base <path = "DIR/a.mojo", content_hash = "807b905a30365194">
  %1 = depgraph.module @dia_left(%0) <path = "DIR/b.mojo", content_hash = "63dd5c36f854c19e">
  %2 = depgraph.module @dia_right(%0) <path = "DIR/b.mojo", content_hash = "63dd5c36f854c19e">
  %3 = depgraph.module @dia_root(%1, %2) <path = "DIR/c.mojo", content_hash = "d9d60135c91335d8">
}

// Two-cycle a <-> b, with top importing a. a changed: reverse reachability is
// total within the cycle, so both members and top are dirty.
// CLEAN-LABEL: depgraph.graph root @cyc_top
// CLEAN-NOT: depgraph.dirty
// DIRTY-LABEL: depgraph.graph root @cyc_top
// DIRTY: depgraph.module @cyc_a{{.*}}depgraph.dirty
// DIRTY: depgraph.module @cyc_b{{.*}}depgraph.dirty
// DIRTY: depgraph.module @cyc_top{{.*}}depgraph.dirty
depgraph.graph root @cyc_top {
  %0 = depgraph.module @cyc_a(%1) <path = "DIR/a.mojo", content_hash = "807b905a30365194">
  %1 = depgraph.module @cyc_b(%0) <path = "DIR/b.mojo", content_hash = "63dd5c36f854c19e">
  %2 = depgraph.module @cyc_top(%0) <path = "DIR/c.mojo", content_hash = "d9d60135c91335d8">
}

// A node whose source file cannot be read is unverifiable, hence dirty, and
// dirtiness reaches its dependent even in the baseline run.
// CLEAN-LABEL: depgraph.graph root @missing_top
// CLEAN: depgraph.module @missing_leaf{{.*}}depgraph.dirty
// CLEAN: depgraph.module @missing_top{{.*}}depgraph.dirty
depgraph.graph root @missing_top {
  %0 = depgraph.module @missing_leaf <path = "DIR/nonexistent.mojo", content_hash = "807b905a30365194">
  %1 = depgraph.module @missing_top(%0) <path = "DIR/c.mojo", content_hash = "d9d60135c91335d8">
}

// A node that was never hashed (no `content_hash`) is dirty even though its
// source file exists.
// CLEAN-LABEL: depgraph.graph root @unhashed
// CLEAN: depgraph.module @unhashed{{.*}}depgraph.dirty
depgraph.graph root @unhashed {
  %0 = depgraph.module @unhashed <path = "DIR/a.mojo">
}
