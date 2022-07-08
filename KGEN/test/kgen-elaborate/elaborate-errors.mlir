// RUN: kgen-elaborate %s -library=%S/library.mlir -verify-diagnostics -o /dev/null -split-input-file

// expected-error @+1 {{interface argument #0 has type 'f32' but library interface expected type 'si32'}}
kgen.generator.interface @unary_add<size>(f32) -> si32


// -----

kgen.generator.interface @genItf2<x>()

kgen.generator @genItf2_impl0<x>()
  constraints <eq(x, 0)> implements @genItf2 {
  "impl0"() : () -> ()
  kgen.return
}
kgen.generator @genItf2_impl1<x>() 
  constraints <eq(x, 1)> implements @genItf2 {
  "impl1"() : () -> ()
  kgen.return
}

// This has no expansions, so it should generate an error message.
// expected-error @+1 {{failed to generate any kernels}}
kgen.kernel @use_Itf2two() {
  kgen.call @genItf2<x = 2>() : () -> ()
  kgen.return
}
