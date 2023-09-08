// RUN: kgen-opt %s -eliminate-dead-symbols -allow-unregistered-dialect | FileCheck %s

// CHECK-NOT: @unused
kgen.func @unused() {
  kgen.return
}

// CHECK: @used
kgen.func @used() {
  kgen.return
}

// CHECK: @addr
kgen.func @addr() {
  kgen.return
}

// CHECK: @someOp
kgen.func @someOp() {
  kgen.return
}

// CHECK: @exported
kgen.func export @exported() {
  kgen.call @used() : () -> ()
  kgen.call @addr() : () -> ()
  "some.op"() {foo=@someOp} : () -> ()
  kgen.return
}

// CHECK: @A
kgen.func export @A() {
  kgen.call @B() : () -> ()
  kgen.return
}

// CHECK: @B
kgen.func @B() {
  kgen.call @A() : () -> ()
  kgen.return
}

// CHECK: @global_var_fn
kgen.func @global_var_fn() {
  kgen.return
}

// CHECK-NOT: @unused_global_fn
kgen.func @unused_global_fn() {
  kgen.return
}

// CHECK: kgen.global @global_var
kgen.global @global_var : i32 [@global_var_fn, @global_var_fn](2)

// CHECK-NOT: kgen.global @global_var
kgen.global @unused_global : i64 [@unused_global_fn, @unused_global_fn](3)

// CHECK: kgen.func export @anchor_global
kgen.func export @anchor_global() {
  kgen.global.address @global_var : <i32>
  kgen.return
}
