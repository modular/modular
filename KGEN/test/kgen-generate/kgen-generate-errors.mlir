// RUN: kgen-generate %s -library=%S/library.mlir -verify-diagnostics -o /dev/null

// expected-error @+1 {{interface argument #0 has type 'f32' but library interface expected type 'si32'}}
kgen.generator.interface @unary_add<size>(f32) -> si32
