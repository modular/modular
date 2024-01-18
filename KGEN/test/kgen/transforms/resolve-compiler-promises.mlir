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

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @call_fn
// CHECK-SAME: (%arg0: !kgen.pointer<none>)
kgen.func @call_fn(%arg0: !kgen.pointer<none>) -> index {
  // CHECK-NEXT: %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index)>>
  // CHECK-NEXT: %1 = kgen.struct.gep %0[0]
  // CHECK-NEXT: %2 = pop.load %1
  kgen.capture_list.expand %arg0
  %0 = pop.compiler.global_load "foo" : index
  // CHECK-NEXT: return %2
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @init_fn
// CHECK-SAME: (%arg0: index)
kgen.func @init_fn() capturing -> !kgen.pointer<none> {
  // CHECK: %0 = pop.aligned_alloc %idx8, %idx8
  // CHECK-NEXT: %1 = kgen.struct.gep %0[0] : <struct<(index)>>
  // CHECK-NEXT: store %arg0, %1
  // CHECK-NEXT: %2 = pop.pointer.bitcast
  %cl = kgen.capture_list.create :(!kgen.pointer<none>) -> index @call_fn
  // CHECK-NEXT: return %2
  kgen.return %cl : !kgen.pointer<none>
}

// CHECK-LABEL: kgen.func @call_it
// CHECK-SAME: (%arg0: !kgen.pointer<none>)
kgen.func @call_it(%arg0: !kgen.pointer<none>) {
  // CHECK-NEXT: call @call_fn(%arg0)
  kgen.call @call_fn(%arg0) : (!kgen.pointer<none>) -> index
  kgen.return
}

// CHECK-LABEL: kgen.func @copy_fn
// CHECK-SAME: (%arg0: !kgen.pointer<none>)
kgen.func @copy_fn(%arg0: !kgen.pointer<none>) -> !kgen.pointer<none> {
  // CHECK: %0 = pop.aligned_alloc %idx8, %idx8
  // CHECK-NEXT: %1 = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<struct<(index)>>
  // CHECK-NEXT: %2 = pop.load %1
  // CHECK-NEXT: pop.store %2, %0
  // CHECK-NEXT: %3 = pop.pointer.bitcast %0
  %ptr = kgen.capture_list.copy %arg0 :(!kgen.pointer<none>) -> index @call_fn
  // CHECK-NEXT: return %3
  kgen.return %ptr : !kgen.pointer<none>
}

}

// -----

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="p:64:64", simd_bit_width=128>} {

// CHECK-LABEL: kgen.func @init_fn
kgen.func @init_fn(%arg0: i64, %arg1: i32) capturing -> !kgen.pointer<none> {
  // CHECK: store %arg0
  // CHECK: store %arg1
  // CHECK: bitcast %{{.*}} : !kgen.pointer<struct<(i64, i32)>>
  %cl = kgen.capture_list.create :(!kgen.pointer<none>) -> () @call_fn
  kgen.return %cl : !kgen.pointer<none>
}

kgen.func @closure1() capturing {
  %0 = pop.compiler.global_load "cap1" : i32
  kgen.return
}

kgen.func @closure2() capturing {
  %0 = pop.compiler.global_load "cap2" : i64
  kgen.return
}

kgen.func @join() capturing {
  kgen.call @closure1() : () capturing -> ()
  kgen.call @closure2() : () capturing -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func @call_fn
// CHECK-SAME: (%arg0: !kgen.pointer<none>)
kgen.func @call_fn(%cl: !kgen.pointer<none>) {
  // CHECK: [[C0:%.*]] = pop.load
  // CHECK: [[C1:%.*]] = pop.load
  kgen.capture_list.expand %cl
  // CHECK: call @join([[C0]], [[C1]])
  kgen.call @join() : () capturing -> ()
  kgen.return
}

}
