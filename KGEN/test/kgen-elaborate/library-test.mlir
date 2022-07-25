// This is the library of kernel generators used for infrastructure tests.
// It is also run as a test, which just verifies that it parses correctly.

// RUN: kgen-opt %s | kgen-opt -o /dev/null

// CHECK-LABEL: kgen.generator.interface @unary_add<size>(si32) -> si32

// expected-note @+1 {{library interface}}
kgen.generator.interface @unary_add<size>(si32) -> si32

// Trivial kernel so we can call something
kgen.kernel @unary_add_library_impl() {
  kgen.return
}

kgen.generator @unary_add_library_impl1<size>(%arg0: si32) -> si32
  implements @unary_add {

  // Silly kernel so we know when something used this.
  kgen.call @unary_add_library_impl() : () -> ()

  // TODO: Do something with <size>
  kgen.return %arg0 : si32
}
