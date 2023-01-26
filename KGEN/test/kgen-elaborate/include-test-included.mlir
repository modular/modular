// This file is part of include-test.mlir, just make sure it parses correctly
// RUN: kgen-opt %s -allow-unregistered-dialect

kgen.include "library-test.mlir"

kgen.generator.interface @genItf2<x>()

kgen.generator @genItf2_impl0<x>()
  constraints <[eq(x, 0), "x must be zero"]> implements @genItf2 {
  "impl.0"() : () -> ()
  kgen.return
}

kgen.generator @genItf2_impl1<x>()
  constraints <[eq(x, 1), "x must be 1"]> implements @genItf2 {
  "impl.1"() : () -> ()
  kgen.return
}
