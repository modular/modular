// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

lit.struct.decl @MyStruct<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
// CHECK-SAME: !kgen.declref<@MyStruct<a = a, b: dtype = b, c: type = c>>
kgen.generator @UseStruct<a, b: dtype, c: type>(%arg0: !kgen.declref<@MyStruct<a = a, b: dtype = b, c: type = c>>) {
  kgen.return
}

// CHECK-LABEL: @sugaredScalar
// CHECK-SAME: %arg0: !pop.pointer<*"scalar">
kgen.generator @sugaredScalar<scalar: type>(%arg0: !pop.pointer<*"scalar">) {
  kgen.return
}

// CHECK-LABEL: @int_literal
// CHECK-SAME: %arg0: !kgen.int_literal
kgen.func @int_literal(%arg0: !kgen.int_literal) {
  kgen.return
}
