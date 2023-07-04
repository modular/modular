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
kgen.func @exported() {
  kgen.call @used() : () -> ()
  kgen.call @addr() : () -> ()
  "some.op"() {foo=@someOp} : () -> ()
  kgen.return
}

kgen.export @exported

// CHECK: @A
kgen.func @A() {
  kgen.call @B() : () -> ()
  kgen.return
}

// CHECK: @B
kgen.func @B() {
  kgen.call @A() : () -> ()
  kgen.return
}

kgen.export @A

// CHECK: @global_var_ctor
kgen.func @global_var_ctor() {
  kgen.return
}

// CHECK: @global_var_dtor
kgen.func @global_var_dtor() {
  kgen.return
}

// CHECK: kgen.global @global_var {{.*}} @global_var_ctor, @global_var_dtor
kgen.global @global_var : i32 (2, @global_var_ctor, @global_var_dtor)
