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
  %struct = lit.struct.create(a=%a, b=%b) : (index, index) -> !lit.declref<@FooStruct>
  %field = lit.struct.extract %struct[a] : index from !lit.declref<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_create_b
lit.func @struct_extract_fold_create_b(%a: index, %b: index) -> index {
  // CHECK-NOT: lit.struct.create
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %b
  %struct = lit.struct.create(a=%a, b=%b) : (index, index) -> !lit.declref<@FooStruct>
  %field = lit.struct.extract %struct[b] : index from !lit.declref<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_fold_insert
lit.func @struct_extract_fold_insert(%struct0: !lit.declref<@FooStruct>) -> index {
  // CHECK-NOT: lit.struct.insert
  // CHECK-NOT: lit.struct.extract
  // CHECK: kgen.return %idx10
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !lit.declref<@FooStruct>
  %field = lit.struct.extract %struct1[a] : index from !lit.declref<@FooStruct>
  kgen.return %field : index
}

// CHECK-LABEL: lit.func @struct_extract_no_fold_insert
lit.func @struct_extract_no_fold_insert(%struct0: !lit.declref<@FooStruct>) -> index {
  // CHECK: lit.struct.insert
  // CHECK-NEXT: lit.struct.extract
  // CHECK-NEXT: kgen.return
  %x = index.constant 10
  %struct1 = lit.struct.insert %x, %struct0[a] : index into !lit.declref<@FooStruct>
  %field = lit.struct.extract %struct1[b] : index from !lit.declref<@FooStruct>
  kgen.return %field : index
}

lit.func @struct_ops_fold() -> (!lit.declref<@FooStruct>, !lit.declref<@FooStruct>, index) {
  // CHECK-DAG: %[[V0:.*]] = {{.*}} @FooStruct = <{a = 0, b = 0}>
  // CHECK-DAG: %[[V1:.*]] = {{.*}} @FooStruct = <{a = 0, b = 3}>
  // CHECK-DAG: %[[V2:.*]] = {{.*}} = <3>
  %idx0 = index.constant 0
  %0 = lit.struct.create(a=%idx0, b=%idx0) : (index, index) -> !lit.declref<@FooStruct>

  %1 = kgen.param.constant: !lit.declref<@FooStruct> = <#lit.struct<{a = 2, b = 3}>>
  %2 = lit.struct.insert %idx0, %1[a] : index into !lit.declref<@FooStruct>

  %3 = lit.struct.extract %1[b] : index from !lit.declref<@FooStruct>

  // CHECK: return %[[V0]], %[[V1]], %[[V2]]
  kgen.return %0, %2, %3 : !lit.declref<@FooStruct>, !lit.declref<@FooStruct>, index
}

lit.struct.decl @Pair register_passable_trivial {
  lit.struct.field first : !lit.declref<@Int>
  lit.struct.field second : !lit.declref<@Int>
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

// -----

// COM: Check that constant are only hoisted from subprogram regions if there is
// COM: no debuginfo scope given.

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (): DW_CC_normal>
#subprogram1 = #debuginfo.subprogram<name = <"SomeClosure">> : !debuginfo.subroutine<() -> (): DW_CC_normal>

#loc1 = loc("foo.mlir":44:1)
#loc2 = loc("foo.mlir":325:11)
#loc3 = loc("bar.mlir":327:17)
#loc4 = loc(fused<#subprogram>[#loc1])
#loc5 = loc(fused<#subprogram1>[#loc2])
#loc6 = loc(fused<#subprogram1>[#loc3])
#call_loc = #debuginfo.call_loc<#loc4>
#loc7 = loc(fused<#call_loc>[#loc2])
#loc8 = loc(fused<#subprogram1>[#loc7])

// CHECK-LABEL: kgen.func @no_hoist
kgen.func @no_hoist() -> !pop.coroutine<() -> ()> {
  // CHECK-NEXT: lit.async.execute <() -> ()> {
  %0 = lit.async.execute <() -> ()> {
    // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]> loc(#loc6)
    %1 = pop.stack_allocation 1 x !pop.array<1, index>  loc(#loc6)
    pop.store %array, %1 : !kgen.pointer<array<1, index>> loc(#loc6)
    kgen.return  loc(#loc5)
  } loc(#loc8)
  kgen.return %0 : !pop.coroutine<() -> ()> loc(#loc4)
} loc(#loc4)

// CHECK-LABEL: kgen.func @hoist
kgen.func @hoist() -> !pop.coroutine<() -> ()> {
  // CHECK-NEXT: kgen.param.constant: array<1, index> = <[0]>
  // CHECK-NEXT: lit.async.execute <() -> ()> {
  %0 = lit.async.execute <() -> ()> {
    // CHECK-NOT: kgen.param.constant: array<1, index> = <[0]>
    %array = kgen.param.constant: array<1, index> = <[0]>
    %1 = pop.stack_allocation 1 x !pop.array<1, index>
    pop.store %array, %1 : !kgen.pointer<array<1, index>>
    kgen.return
  }
  kgen.return %0 : !pop.coroutine<() -> ()>
}

// CHECK-LABEL: @no_cse_async_execute
kgen.func @no_cse_async_execute() -> (!pop.coroutine<() -> ()>, !pop.coroutine<() -> ()>) {
  // CHECK-COUNT-2: lit.async.execute
  %0 = lit.async.execute <() -> ()> {
    kgen.return
  }
  %1 = lit.async.execute <() -> ()> {
    kgen.return
  }
  kgen.return %0, %1 : !pop.coroutine<() -> ()>, !pop.coroutine<() -> ()>
}
