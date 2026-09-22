// RUN: kgen-opt %s -eliminate-dead-symbols -allow-unregistered-dialect | FileCheck %s

// CHECK-NOT: @unused
kgen.func @unused() {
  hlcf.return
}

// CHECK: @used
kgen.func @used() {
  hlcf.return
}

// CHECK: @addr
kgen.func @addr() {
  hlcf.return
}

// CHECK: @someOp
kgen.func @someOp() {
  hlcf.return
}

// CHECK: @exported
kgen.func export @exported() {
  kgen.call @used() : () -> ()
  kgen.call @addr() : () -> ()
  "some.op"() {foo=@someOp} : () -> ()
  hlcf.return
}

// CHECK: @A
kgen.func export @A() {
  kgen.call @B() : () -> ()
  hlcf.return
}

// CHECK: @B
kgen.func @B() {
  kgen.call @A() : () -> ()
  hlcf.return
}
