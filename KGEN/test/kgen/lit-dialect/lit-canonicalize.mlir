// RUN: kgen-opt -canonicalize -mlir-print-debuginfo -split-input-file %s | FileCheck %s

// This shouldn't crash.
// https://github.com/modularml/modular/issues/2480

lit.struct.decl @FooStruct {
  lit.struct.field a : index
  lit.struct.field b : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_create
lit.func @struct_extract_fold_create(%a: index, %b: index) -> index {
  // CHECK-NOT: lit.struct.create
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %a
  %struct = lit.struct.create(a=%a, b=%b) : (index, index) -> !lit.struct<@FooStruct>
  %field = lit.struct.extract %struct[a] : index from !lit.struct<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_create_b
lit.func @struct_extract_fold_create_b(%a: index, %b: index) -> index {
  // CHECK-NOT: lit.struct.create
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %b
  %struct = lit.struct.create(a=%a, b=%b) : (index, index) -> !lit.struct<@FooStruct>
  %field = lit.struct.extract %struct[b] : index from !lit.struct<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_insert
lit.func @struct_extract_fold_insert(%struct0: !lit.struct<@FooStruct>) -> index {
  // CHECK-NOT: lit.struct.insert
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %idx10
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !lit.struct<@FooStruct>
  %field = lit.struct.extract %struct1[a] : index from !lit.struct<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_no_fold_insert
lit.func @struct_extract_no_fold_insert(%struct0: !lit.struct<@FooStruct>) -> index {
  // CHECK: lit.struct.insert
  // CHECK-NEXT: lit.struct.extract
  // CHECK-NEXT: kgen.return
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !lit.struct<@FooStruct>
  %field = lit.struct.extract %struct1[b] : index from !lit.struct<@FooStruct>
  kgen.return %field : index
}

lit.func @struct_ops_fold() -> (!lit.struct<@FooStruct>, !lit.struct<@FooStruct>, index) {
  // CHECK-DAG: %[[V0:.*]] = {{.*}} @FooStruct = <{a = 0, b = 0}>
  // CHECK-DAG: %[[V1:.*]] = {{.*}} @FooStruct = <{a = 0, b = 3}>
  // CHECK-DAG: %[[V2:.*]] = {{.*}} = <3>
  %idx0 = index.constant 0
  %0 = lit.struct.create(a=%idx0, b=%idx0) : (index, index) -> !lit.struct<@FooStruct>

  %1 = kgen.param.constant: !lit.struct<@FooStruct> = <#lit.struct<{a = 2, b = 3}>>
  %2 = lit.struct.insert %idx0, %1[a] : index into !lit.struct<@FooStruct>

  %3 = lit.struct.extract %1[b] : index from !lit.struct<@FooStruct>

  // CHECK: return %[[V0]], %[[V1]], %[[V2]]
  kgen.return %0, %2, %3 : !lit.struct<@FooStruct>, !lit.struct<@FooStruct>, index
}

lit.struct.decl @Pair register_passable_trivial {
  lit.struct.field first : !lit.struct<@Int>
  lit.struct.field second : !lit.struct<@Int>
}

lit.struct.decl @Int register_passable_trivial {
  lit.struct.field value : index
}


// CHECK-LABEL: lit.func @fold_ger
lit.func @fold_ger[mut lt]() -> !lit.ref<index, mut lt> {
  // CHECK-NEXT: kgen.param.constant: !lit.ref<index, mut lt> = <#lit.struct.ger<#lit.struct.ger<#interp.symbolic_pointer<0> : !lit.ref<@Pair, mut lt>, "first"> : !lit.ref<@Int, mut lt>, "value">>
  %x = kgen.param.constant: !lit.ref<@Pair, mut lt> = <#interp.symbolic_pointer<0>>
  %0 = lit.ref.struct.ger %x[first] : <@Int, mut lt> from @Pair
  %1 = lit.ref.struct.ger %0[value] : <index, mut lt> from @Int
  kgen.return %1 : !lit.ref<index, mut lt>
}
