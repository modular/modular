// RUN: kgen-opt %s | kgen-opt | FileCheck %s

kgen.struct.decl @MyStruct<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
kgen.generator.interface @UseStruct<a, b: dtype, c: type>() ->
    // CHECK-SAME: !kgen.ref<@MyStruct<a = a, b: dtype = b, c: type = c>>
    !kgen.ref<@MyStruct<a = a, b: dtype = b, c: type = c>>
