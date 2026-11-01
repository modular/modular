// RUN: mo-opt %s  --mogg-to-mgp | FileCheck %s
// Test invoking `mo-to-primitives` removes any lit register ops.

mo.graph @empty() -> () {
  mo.output
}


// CHECK-NOT: mogg.lit_register
