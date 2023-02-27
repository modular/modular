// RUN: kgen-opt %s -split-input-file -elaborate-generators="enable-search=true" -allow-unregistered-dialect | FileCheck %s

// CHECK: kgen.func @"recurse,axis=2"(%arg0: index) -> index {
// CHECK-NEXT:    %0 = kgen.call @"recurse,axis=1"(%arg0) : (index) -> index
// CHECK-NEXT:    kgen.return %0 : index

// CHECK: kgen.func @"recurse,axis=1"(%arg0: index) -> index {
// CHECK-NEXT:    %0 = kgen.call @"recurse,axis=0"(%arg0) : (index) -> index
// CHECK-NEXT:    kgen.return %0 : index

// CHECK: kgen.func @"recurse,axis=0"(%arg0: index) -> index {
// CHECK-NEXT:    kgen.return %arg0 : index

// CHECK:    %1 = kgen.call @"recurse,axis=2"(%idx42) : (index) -> index

module {
  kgen.generator @recurse<axis>(%arg0: index) -> index {
	kgen.param.if <eq(axis, 0)> {
  	   hlcf.return %arg0 : index
	} else {
  	   kgen.param.yield
	}
	%0 = kgen.call @recurse<axis = add(axis, -1)>(%arg0) : (index) -> index
	kgen.return %0 : index
  }
  kgen.generator @main() -> !kgen.list<i1[0]> {
	%0 = kgen.param.constant: list<i1[0]> = <[]>
	%idx42 = index.constant 42
	%1 = kgen.call @recurse<axis = 2>(%idx42) : (index) -> index
	kgen.return %0 : !kgen.list<i1[0]>
  }
  kgen.export @main  as @main
}
