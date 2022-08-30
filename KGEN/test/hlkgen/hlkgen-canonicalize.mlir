// RUN: kgen-opt -canonicalize %s | FileCheck %s

// This shouldn't crash.
// https://github.com/modularml/modular/issues/2480

// CHECK-LABEL: kgen.generator.interface @numeric_limits.digits
kgen.generator.interface @numeric_limits.digits<type: dtype -> digits>()
// CHECK-LABEL: hlkgen.generator @numeric_limits.digits.i32
// CHECK-NEXT: constraints <[eq(:dtype type, si32), "this only works for si32", #
// CHECK-NEXT: implements @numeric_limits.digits {
hlkgen.generator @numeric_limits.digits.i32<type: dtype -> digits>()
    implements @numeric_limits.digits {
  kgen.param.assert <eq(:dtype type, si32)>, "this only works for si32"
  kgen.return <digits = 31>
}
