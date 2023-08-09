// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

lit.struct.decl @MyStruct<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
// CHECK-SAME: !kgen.declref<@MyStruct<a = a, b: dtype = b, c: type = c>>
kgen.generator @UseStruct<a, b: dtype, c: type>(%arg0: !kgen.declref<@MyStruct<a = a, b: dtype = b, c: type = c>>) {
  kgen.return
}

// CHECK: !kgen.int_literal
kgen.func @int_literal(%arg0: !kgen.int_literal) {
  kgen.return
}
