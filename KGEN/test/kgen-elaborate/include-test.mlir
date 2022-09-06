// RUN: kgen-opt %s -elaborate-generators="search-path=%S" -allow-unregistered-dialect | FileCheck %s

kgen.include "elaborate.mlir"

kgen.generator.interface @genItf2<x>()
kgen.generator.interface @unary_add<size>(si32) -> si32

// CHECK-LABEL:  kgen.kernel @"genItf2_impl0,x=0"() {

// CHECK-LABEL: kgen.kernel @useAnInclude
// CHECK-NEXT:    kgen.call @"genItf2_impl0,x=0"
kgen.generator @useAnInclude(%arg0: si32) -> si32 {
  kgen.call @genItf2<x = 0>() : () -> ()
  kgen.return %arg0 : si32
}

// This is from the generator below - it calls this kernel but we clone it
// into this to be self contained.

// CHECK-LABEL: kgen.kernel @unary_add_library_impl

// CHECK-LABEL: kgen.kernel @"unary_add_library_impl1,size=1"
// CHECK-NEXT:    kgen.call @unary_add_library_impl

// CHECK-LABEL: kgen.kernel @useANestedInclude
// CHECK-NEXT:    kgen.call @"unary_add_library_impl1,size=1"
kgen.generator @useANestedInclude(%arg0: si32) -> si32 {
  %0 = kgen.call @unary_add<size = 1>(%arg0) : (si32) -> si32
  kgen.return %0 : si32
}
