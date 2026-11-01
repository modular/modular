// RUN: mt --execute %s | FileCheck %s

// CHECK: --- Running 'constant_bool':
// CHECK-NEXT: 'constant_bool' returned 1
mo.graph @constant_bool() -> !mo.scalar<bool> {
  %0 = mo.constant.scalar <1 : i1> : !mo.scalar<bool>
  mo.output %0 : !mo.scalar<bool>
}

// CHECK: --- Running 'constant_si32':
// CHECK-NEXT: 'constant_si32' returned 20
mo.graph @constant_si32() -> !mo.scalar<si32> {
  %0 = mo.constant.scalar <20 : si32> : !mo.scalar<si32>
  mo.output %0 : !mo.scalar<si32>
}
