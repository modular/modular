// RUN: kgen-opt %s -split-input-file -verify-parameters -resolve-compiler-promises -canonicalize -verify-parameters -o %t
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
  // CHECK: call @transitive(%arg0, %arg1) : (i32, index) capturing -> ()
  kgen.call @transitive() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @transitive(%arg0: i32, %arg1: index)
kgen.func @transitive() capturing {
  // CHECK: call @inner(%arg1)
  kgen.call @inner() : () capturing -> ()
  %0 = pop.compiler.global_load "baz" : i32
  // CHECK: call @use_i32(%arg0)
  kgen.call @use_i32(%0) : (i32) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @inner(%arg0: index)
kgen.func @inner() capturing {
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
kgen.func @make_a_closure(%arg0: index) -> !kgen.signature<()capturing -> ()> {
  pop.compiler.global_store "foobar", %arg0 : index
  // CHECK: create_closure[(index) capturing -> (): @capturing](%arg0)
  %0 = kgen.create_closure[() capturing -> (): @capturing]()
  kgen.return %0: !kgen.signature<()capturing -> ()>
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
  // CHECK: lit.async.call[(index) async|capturing -> (): @async_fn](%arg0)
  lit.async.call [() async|capturing -> (): @async_fn]()
  kgen.return
}

// CHECK-LABEL: kgen.func @async_fn(%arg0: index) async
kgen.func @async_fn() async|capturing {
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

// COM: The cyclic node does not form an edge in the call graph.

// CHECK-LABEL: kgen.func @scc_pred(%arg0: index)
kgen.func @scc_pred(%arg0: index) {
  pop.compiler.global_store "var", %arg0 : index
  // CHECK: call @request(%arg0)
  kgen.call @request() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @request(%arg0: index)
kgen.func @request() capturing {
  pop.compiler.global_load "var" : index
  // CHECK: call @cyclic() : () -> ()
  kgen.call @cyclic() : () -> ()
  kgen.return
}

kgen.func @cyclic() {
  kgen.call @cyclic() : () -> ()
  kgen.return
}

// COM: Just don't crash.

kgen.extern.func @external() -> ()

kgen.func @call_external() {
  kgen.call @external() : () -> ()
  kgen.return
}

// -----

!capture_list_type = !kgen.capture_list<() capturing -> (): @my_capture>
!capture_data_type = !kgen.struct<(index)>

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {
// CHECK-LABEL: kgen.func @my_capture(%arg0: index) capturing
kgen.func @my_capture() capturing {
  // CHECK-NOT: pop.compiler.global_load
  %0 = pop.compiler.global_load "var0" : index
  kgen.return
}

// CHECK-LABEL: kgen.func @capture_list_create()
kgen.func @capture_list_create() -> !capture_list_type {
  // CHECK-NOT: kgen.capture_list.create
  // CHECK-NEXT: [[IDX8:%.*]] = index.constant 8
  // CHECK-NEXT: [[V0:%.*]] = pop.aligned_alloc [[IDX8]], [[IDX8]] : <struct<(index)>>
  %capture_list = kgen.capture_list.create : !capture_list_type
  kgen.return %capture_list: !capture_list_type
}

// CHECK-LABEL: kgen.func @capture_list_expand(%arg0: !kgen.pointer<struct<(index)>>) {
kgen.func @capture_list_expand(%capture_list: !capture_list_type) {
  // CHECK-NOT: kgen.capture_list.expand
  // CHECK-NEXT: [[V0:%.*]] = kgen.struct.gep %arg0[0] : <struct<(index)>>
  // CHECK-NEXT: [[V1:%.*]] = pop.load [[V0]] : !kgen.pointer<index>
  // CHECK-NEXT: kgen.call @my_capture([[V1]]) : (index) capturing -> ()

  kgen.capture_list.expand %capture_list: !capture_list_type
  kgen.call @my_capture(): () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @capture_list_destroy(%arg0: !kgen.pointer<struct<(index)>>) {
kgen.func @capture_list_destroy(%capture_list: !capture_list_type) {
  // CHECK-NOT: kgen.capture_list.destroy
  // CHECK-NEXT: pop.aligned_free %arg0 : <struct<(index)>>
  kgen.capture_list.destroy %capture_list: !capture_list_type
  kgen.return
}

}
