// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

// Round-trip the depgraph dialect through both the textual and bytecode
// readers/writers to verify the `root` symbol attribute is preserved.

// CHECK-LABEL: depgraph.graph root @main {
// CHECK:   %[[STD:.+]] = depgraph.module @std
// CHECK:   %[[UTIL:.+]] = depgraph.module @util(%[[STD]])
// CHECK:   depgraph.module @main(%[[UTIL]], %[[STD]])
// CHECK: }
depgraph.graph root @main {
  %std  = depgraph.module @std
  %util = depgraph.module @util(%std)
  %main = depgraph.module @main(%util, %std)
}

// A graph with no root is also valid.
// CHECK-LABEL: depgraph.graph {
// CHECK-NOT: root
// CHECK:   depgraph.module @lonely
// CHECK: }
depgraph.graph {
  %lonely = depgraph.module @lonely
}

// A self-cycle is representable in the graph region.
// CHECK-LABEL: depgraph.graph root @cyclic {
// CHECK:   %[[A:.+]] = depgraph.module @cyclic(%[[B:.+]])
// CHECK:   %[[B]] = depgraph.module @b(%[[A]])
// CHECK: }
depgraph.graph root @cyclic {
  %a = depgraph.module @cyclic(%b)
  %b = depgraph.module @b(%a)
}
