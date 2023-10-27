// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

// CHECK-LABEL: @sugaredScalar
// CHECK-SAME: %arg0: !kgen.pointer<*"scalar">
kgen.generator @sugaredScalar<scalar: type>(%arg0: !kgen.pointer<*"scalar">) {
  kgen.return
}

// CHECK-LABEL: @int_literal
// CHECK-SAME: %arg0: !kgen.int_literal
kgen.func @int_literal(%arg0: !kgen.int_literal) {
  kgen.return
}
