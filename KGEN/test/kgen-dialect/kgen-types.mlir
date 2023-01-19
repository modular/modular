// RUN: kgen-opt %s | kgen-opt | FileCheck %s

lit.struct.decl @MyStruct<a, b: dtype, c: type> {}

// CHECK-LABEL: @UseStruct
kgen.generator.interface @UseStruct<a, b: dtype, c: type>() ->
    // CHECK-SAME: !kgen.declref<@MyStruct<a = a, b: dtype = b, c: type = c>>
    !kgen.declref<@MyStruct<a = a, b: dtype = b, c: type = c>>
