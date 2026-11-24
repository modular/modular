// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -verify-diagnostics %s

// A `root` that does not name a module in the graph is rejected.
// expected-error @+1 {{'root' symbol 'missing' does not name a 'depgraph.module' in this graph}}
depgraph.graph root @missing {
  %std = depgraph.module @std
}

// -----

// Only `depgraph.module` ops may appear inside a graph.
depgraph.graph {
  // expected-error @+1 {{only 'depgraph.module' ops are allowed inside a 'depgraph.graph'}}
  "test.unknown"() : () -> ()
}
