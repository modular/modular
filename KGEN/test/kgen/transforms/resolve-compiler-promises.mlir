// RUN: kgen-opt %s -resolve-compiler-promises -verify-parameters -o %t
// RUN: cat %t | FileCheck %s --check-prefix=GONE
// RUN: cat %t | FileCheck %s

// GONE: kgen.func @use
// GONE-NONE: pop.compiler

kgen.func @use(%arg0: index) {
  kgen.unreachable
}

kgen.func @use_i32(%arg0: i32) {
  kgen.unreachable
}

// CHECK-LABEL: kgen.func @top(%arg0: i32, %arg1: index)
kgen.func @top(%arg0: index) {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK: call @transitive(%arg0, %arg1) : (i32, index) -> ()
  kgen.call @transitive() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @transitive(%arg0: i32, %arg1: index)
kgen.func @transitive() {
  // CHECK: call @inner(%arg1)
  kgen.call @inner() : () -> ()
  %0 = pop.compiler.global_load "baz" : i32
  // CHECK: call @use_i32(%arg0)
  kgen.call @use_i32(%0) : (i32) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @inner(%arg0: index)
kgen.func @inner() {
  %idx0 = index.constant 0
  pop.compiler.global_store "index", %idx0 : index
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg0)
  kgen.call @use(%0) : (index) -> ()
  %1 = pop.compiler.global_load "index" : index
  // CHECK: call @use(%idx0)
  kgen.call @use(%1) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @make_a_closure(%arg0: index)
kgen.func @make_a_closure(%arg0: index) {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK: create_closure [(index) capturing -> (): @capturing](%arg0)
  kgen.create_closure [() capturing -> (): @capturing]()
  kgen.return
}

// CHECK-LABEL: kgen.func @capturing(%arg0: index) capturing
kgen.func @capturing() capturing {
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg0)
  kgen.call @use(%0) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @make_a_coroutine(%arg0: index)
kgen.func @make_a_coroutine(%arg0: index) {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK: lit.async.call[(index) async -> (): @async_fn](%arg0)
  lit.async.call [() async -> (): @async_fn]()
  kgen.return
}

// CHECK-LABEL: kgen.func @async_fn(%arg0: index) async
kgen.func @async_fn() async {
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg0)
  kgen.call @use(%0) : (index) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @unused_store
kgen.func @unused_store(%arg0: index) {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK-NEXT: kgen.return
  kgen.return
}

// CHECK-LABEL kgen.func @store_does_not_dominate(%arg0: index, %arg1: index)
kgen.func @store_does_not_dominate(%arg0: index) {
  // CHECK: loop
  hlcf.loop {
    pop.compiler.global_store "foobar", %arg0 : index
    // CHECK: call @use(%arg1)
    kgen.call @use(%arg0) : (index) -> ()
    // CHECK: break
    hlcf.break
  }
  %0 = pop.compiler.global_load "foobar" : index
  // CHECK: call @use(%arg0)
  kgen.call @use(%0) : (index) -> ()
  kgen.return
}
