// RUN: kgen-opt -canonicalize %s | FileCheck %s

// This shouldn't crash.
// https://github.com/modularml/modular/issues/2480

// CHECK-LABEL: kgen.generator.interface @numeric_limits.digits
kgen.generator.interface @numeric_limits.digits<type: dtype -> index>()
// CHECK-LABEL: lit.func @numeric_limits.digits.i32
// CHECK-NEXT: constraints <[eq(:dtype type, si32), "this only works for si32", #
// CHECK-NEXT: implements @numeric_limits.digits {
lit.func @numeric_limits.digits.i32<type: dtype -> index>()
    implements @numeric_limits.digits {
  kgen.param.assert <eq(:dtype type, si32)>, "this only works for si32"
  kgen.return<31>
}

lit.struct.decl @FooStruct {
  lit.struct.field a : index
  lit.struct.field b : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_create
lit.func @struct_extract_fold_create(%a: index, %b: index) -> index {
  // CHECK-NOT: lit.struct.create
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %a
  %struct = lit.struct.create(a=%a, b=%b) : (index, index) -> !kgen.declref<@FooStruct>
  %field = lit.struct.extract %struct[a] : index from !kgen.declref<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_create_b
lit.func @struct_extract_fold_create_b(%a: index, %b: index) -> index {
  // CHECK-NOT: lit.struct.create
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %b
  %struct = lit.struct.create(a=%a, b=%b) : (index, index) -> !kgen.declref<@FooStruct>
  %field = lit.struct.extract %struct[b] : index from !kgen.declref<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_insert
lit.func @struct_extract_fold_insert(%struct0: !kgen.declref<@FooStruct>) -> index {
  // CHECK-NOT: lit.struct.insert
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %idx10
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !kgen.declref<@FooStruct>
  %field = lit.struct.extract %struct1[a] : index from !kgen.declref<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_no_fold_insert
lit.func @struct_extract_no_fold_insert(%struct0: !kgen.declref<@FooStruct>) -> index {
  // CHECK: lit.struct.insert
  // CHECK-NEXT: lit.struct.extract
  // CHECK-NEXT: kgen.return
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !kgen.declref<@FooStruct>
  %field = lit.struct.extract %struct1[b] : index from !kgen.declref<@FooStruct>
  kgen.return %field : index
}
