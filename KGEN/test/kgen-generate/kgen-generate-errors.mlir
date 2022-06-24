// RUN: not kgen-generate %s -library=%S/library.mlir -verify-diagnostics -o /dev/null

// expected-note @+2 {{see current operation}}
// expected-error @+1 {{interface declared with type (f32) -> si32 but library expects type (si32) -> si32}}
kgen.generator.interface @unary_add<size>(f32) -> si32
