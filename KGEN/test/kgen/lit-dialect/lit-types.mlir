// RUN: kgen-opt %s -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

lit.struct.decl @MyStruct<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
// CHECK-SAME: !kgen.declref<@MyStruct<a, :dtype b, :type c>>
kgen.generator @UseStruct<a, b: dtype, c: type>(%arg0: !kgen.declref<@MyStruct<a, :dtype b, :type c>>) {
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

// CHECK: !lit.type_signature
"type.sig"() : () -> !lit.type_signature
// CHECK: !lit.type_signature<index, |>
"type.sig"() : () -> !lit.type_signature<index, |>
// CHECK: !lit.type_signature<"dt": dtype = f32>
"type.sig"() : () -> !lit.type_signature<"dt": dtype = f32>
// CHECK: !lit<type_signature<"i": variadic<index>> param_vararg>
"type.sig"() : () -> !lit<type_signature<"i": variadic<index>> param_vararg>
