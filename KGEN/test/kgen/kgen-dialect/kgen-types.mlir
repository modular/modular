// RUN: kgen-opt %s | kgen-opt | FileCheck %s
// RUN: kgen-opt -emit-bytecode %s | kgen-opt | FileCheck %s

lit.struct.decl @MyStruct<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
// CHECK-SAME: !kgen.declref<@MyStruct<a, :dtype b, :type c>>
kgen.generator @UseStruct<a, b: dtype, c: type>(%arg0: !kgen.declref<@MyStruct<a, :dtype b, :type c>>) {
  kgen.return
}

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

// CHECK-LABEL: @metatype
// CHECK-SAME: !kgen.pointer<@MyStruct : metatype<@MyStruct>>
kgen.generator @metatype(%arg0: !kgen.pointer<@MyStruct : !lit.metatype<@MyStruct>>) {
  kgen.return
}

// CHECK-LABEL: @declref_metatype
// CHECK-SAME: !kgen.declref<@MyStruct, !lit.metatype<@MyStruct>>
kgen.generator @declref_metatype(%arg0: !kgen.declref<@MyStruct, !lit.metatype<@MyStruct>>) {
  kgen.return
}
