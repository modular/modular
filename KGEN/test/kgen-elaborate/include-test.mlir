// RUN: kgen-opt %s -elaborate-generators="search-path=%S" -allow-unregistered-dialect | FileCheck %s

kgen.include "include-test-included.mlir"
kgen.include "struct-test.mlir"

kgen.generator.interface @genItf2<x>()
kgen.generator.interface @unary_add<size>(si32) -> si32

// CHECK-LABEL: kgen.func @"wrapInFooImpl,T=i32"

// CHECK-LABEL:  kgen.func @"genItf2_impl0,x=0"() {

// CHECK-LABEL: kgen.func @"unary_add_library_impl1,size=1"
// CHECK-NEXT:    kgen.call @unary_add_library_impl

// CHECK-LABEL: kgen.func @unary_add_library_impl

// CHECK-LABEL: kgen.func @useAnInclude
// CHECK-NEXT:    kgen.call @"genItf2_impl0,x=0"
kgen.generator @useAnInclude(%arg0: si32) -> si32 {
  kgen.call @genItf2<x = 0>() : () -> ()
  kgen.return %arg0 : si32
}

// This is from the generator below - it calls this kernel but we clone it
// into this to be self contained.

// CHECK-LABEL: kgen.func @useANestedInclude
// CHECK-NEXT:    kgen.call @"unary_add_library_impl1,size=1"
kgen.generator @useANestedInclude(%arg0: si32) -> si32 {
  %0 = kgen.call @unary_add<size = 1>(%arg0) : (si32) -> si32
  kgen.return %0 : si32
}

kgen.generator.interface @wrapInFoo<T:type>(!pop.pointer<T>) -> !kgen.declref<@FooStruct<T:type = T>>

// CHECK-LABEL: @FooStruct
kgen.struct.decl @FooStruct<T:type> {
  kgen.struct.field x : !pop.pointer<T>
}

// CHECK-LABEL: kgen.func @useStruct
kgen.generator @useStruct(%a: !pop.pointer<i32>) -> !kgen.declref<@FooStruct<T:type = i32>>{
  // CHECK: kgen.call @"wrapInFooImpl,T=i32"
  %0 = kgen.call @wrapInFoo<T:type = i32>(%a) : (!pop.pointer<i32>) -> !kgen.declref<@FooStruct<T:type = i32>>
  kgen.return %0 : !kgen.declref<@FooStruct<T:type = i32>>
}
