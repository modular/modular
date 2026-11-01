// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor -verbose %t.mef | FileCheck %s

// Ensure threadpool count can be set.
// RUN: mef-executor -verbose --num-threads=4 %t.mef | FileCheck %s

// TODO(#11695): reenable this
// COM: runtime using malloc allocator, and thread pool work queue
// CHECK-LABEL: --- Running 'test_linear'
grt.testcase @test_linear {
  %ch0 = grt.chain.create ()

  // executed immediately.
  // CHECK-NEXT: constructed vt.value(1)
  // CHECK-NEXT: move constructed vt.value(1)
  // CHECK-NEXT: constructed vt.value(2)
  // CHECK-NEXT: move constructed vt.value(2)
  // CHECK-NEXT: constructed vt.value(3)
  // CHECK-NEXT: move constructed vt.value(3)
  // CHECK-NEXT: destroyed vt.value(1)
  // CHECK-NEXT: destroyed vt.value(2)
  // CHECK-NEXT: print vt.value(3)
  // CHECK-NEXT: destroyed vt.value(3)
  // CHECK-NEXT: constructed vt.value(4)
  // CHECK-NEXT: move constructed vt.value(4)
  // CHECK-NEXT: destroyed vt.value(4)
  // CHECK-NEXT: constructed vt.value(5)
  // CHECK-NEXT: move constructed vt.value(5)
  // CHECK-NEXT: destroyed vt.value(5)

  %v1 = "test.vt.constant"() { value = 1 : si32 } : () -> !vt.value
  %v2 = "test.vt.constant"() { value = 2 : si32 } : () -> !vt.value
  %r = "test.vt.add"(%v1, %v2) : (!vt.value,  !vt.value) -> !vt.value
  "test.vt.print"(%r, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)
  "test.vt.constant"() { value = 4 : si32 } : () -> !vt.value
  "test.vt.constant"() { value = 5 : si32 } : () -> !vt.value

  mef.output
}

mef.callable @no_op(%x: !vt.value) -> !vt.value {
  mef.output %x : !vt.value
}

// CHECK-LABEL: --- Running 'call_no_op'
grt.testcase @call_no_op {
  // This test uses one vt.value(2) that is all of these:
  // - Function argument: passed to no_op.
  // - Function result: returned by no_op.
  // - Op input: passed to grt.invoke.
  // - Op output: returned by grt.invoke.

  // CHECK-NEXT: constructed vt.value(2)
  // CHECK-NEXT: move constructed vt.value(2)
  %c1 = "test.vt.constant"() { value = 2 : si32 } : () -> !vt.value

  // no_op returns its argument, which extends vt.value(2)'s lifetime.
  %x = grt.invoke @no_op(%c1) : (!vt.value) -> !vt.value

  %ch0 = grt.chain.create ()

  // CHECK-NEXT: print vt.value(2)
  "test.vt.print"(%x, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)
  // CHECK-NEXT: destroyed vt.value(2)
  mef.output
}

mef.callable @add_one(%x: !vt.value) -> !vt.value {
  %v1 = "test.vt.constant"() { value = 1 : si32 } : () -> !vt.value
  %r = "test.vt.add"(%v1, %x) : (!vt.value,  !vt.value) -> !vt.value
  mef.output %r : !vt.value
}

// CHECK-LABEL: --- Running 'caller'
grt.testcase @caller {
  // CHECK-NEXT: constructed vt.value(3)
  // CHECK-NEXT: move constructed vt.value(3)
  %c1 = "test.vt.constant"() { value = 3 : si32 } : () -> !vt.value

  // CHECK-NEXT: constructed vt.value(1)
  // CHECK-NEXT: move constructed vt.value(1)

  // CHECK-NEXT: constructed vt.value(4)
  // CHECK-NEXT: move constructed vt.value(4)
  // CHECK-NEXT: destroyed vt.value(1)

  %x = grt.invoke @add_one(%c1) : (!vt.value) -> !vt.value

  // TODO: If calls were non-strict, then 3 could be destroyed at the add,
  // before the return.
  // CHECK-NEXT: destroyed vt.value(3)

  %ch0 = grt.chain.create ()

  // CHECK-NEXT: print vt.value(4)
  "test.vt.print"(%x, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)
  // CHECK-NEXT: destroyed vt.value(4)
  mef.output
}
// COM: Because the order of the `Initializing` messages is different locally
// COM: vs on CI, just check for the word `Initializing`.

// CHECK-LABEL: --- Running 'test_hello'
grt.testcase @test_hello {
 // CHECK: hello mef executor!
  %ch0 = grt.chain.create ()
  %ch1 = "test.print_hello"(%ch0) : (!mef.chain) -> !mef.chain
  mef.output
}

// CHECK-LABEL: --- Running 'share_to_two'
grt.testcase @share_to_two {
  // CHECK-NEXT: constructed vt.value(1)
  // CHECK-NEXT: move constructed vt.value(1)
  %c1 = "test.vt.constant"() { value = 1 : si32 } : () -> !vt.value

  %x1, %x2 = "test.vt_share_to_two"(%c1) : (!vt.value) -> (!vt.value, !vt.value)

  %ch0 = grt.chain.create ()

  // CHECK-NEXT: print vt.value(1)
  "test.vt.print"(%x1, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)

  // CHECK-NEXT: print vt.value(1)
  "test.vt.print"(%x2, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)

  // CHECK-NEXT: destroyed vt.value(1)
  mef.output
}

// Testing calling a mef function that forwards its argument into two results.

mef.callable @call_share_to_two(%x: !vt.value) -> (!vt.value, !vt.value) {
  %x1, %x2 = "test.vt_share_to_two_2"(%x) : (!vt.value) -> (!vt.value, !vt.value)
  mef.output %x1, %x2 : !vt.value, !vt.value
}

// CHECK-LABEL: --- Running 'share_to_two_caller'
grt.testcase @share_to_two_caller {
  // CHECK-NEXT: constructed vt.value(1)
  // CHECK-NEXT: move constructed vt.value(1)
  %c1 = "test.vt.constant"() { value = 1 : si32 } : () -> !vt.value

  %x1, %x2 = grt.invoke @call_share_to_two(%c1) : (!vt.value) -> (!vt.value, !vt.value)

  %ch0 = grt.chain.create ()

  // CHECK-NEXT: print vt.value(1)
  "test.vt.print"(%x1, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)

  // CHECK-NEXT: print vt.value(1)
  "test.vt.print"(%x2, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)

  // CHECK-NEXT: destroyed vt.value(1)
  mef.output
}

mef.callable @repeat_n_inc(%n: si32) -> !mef.chain {
  // Returns 1 + %n by initializing %x to 1 and incrementing in a loop.
  %ch0 = grt.chain.create ()
  %x = "test.vt.constant"() { value = 1 : si32 } : () -> !vt.value
  %y = grt.repeat.i32 %n (%x: !vt.value) {
    %v1 = "test.vt.constant"() { value = 1 : si32 } : () -> !vt.value
    %r = "test.vt.add"(%v1, %x) : (!vt.value,  !vt.value) -> !vt.value
    mef.output %r : !vt.value
  }

  %ch_out = "test.vt.print"(%y, %ch0) : (!vt.value, !mef.chain) -> (!mef.chain)
  mef.output %ch_out : !mef.chain
}

// CHECK-LABEL: --- Running 'repeat_0_inc'
grt.testcase @repeat_0_inc {
  %count = grt.constant.si32 0

  // CHECK-NEXT: constructed vt.value(1)
  // CHECK-NEXT: move constructed vt.value(1)
  // CHECK-NEXT: print vt.value(1)

  // Destruction of the final value runs async in parallel with the mef-executor
  // printing of the output chain, and can therefore be interlaced anywhere in
  // the subsequent output.
  %ch_out = grt.invoke @repeat_n_inc(%count) : (si32) -> !mef.chain
  mef.output %ch_out : !mef.chain
}

// CHECK-LABEL: --- Running 'repeat_1_inc'
grt.testcase @repeat_1_inc {
  %count = grt.constant.si32 1

  // CHECK: constructed vt.value(1)
  // CHECK: move constructed vt.value(1)
  // CHECK: constructed vt.value(1)
  // CHECK: move constructed vt.value(1)
  // CHECK: constructed vt.value(2)
  // CHECK: move constructed vt.value(2)
  // CHECK: print vt.value(2)

  // Destruction of the final value runs async in parallel with the mef-executor
  // printing of the output chain, and can therefore be interlaced anywhere in
  // the subsequent output.
  %ch_out = grt.invoke @repeat_n_inc(%count) : (si32) -> !mef.chain
  mef.output %ch_out : !mef.chain
}

// CHECK-LABEL: --- Running 'repeat_2_inc'
grt.testcase @repeat_2_inc {
  %count = grt.constant.si32 2

  // CHECK: constructed vt.value(1)
  // CHECK: move constructed vt.value(1)
  // CHECK: constructed vt.value(1)
  // CHECK: move constructed vt.value(1)
  // CHECK: constructed vt.value(2)
  // CHECK: move constructed vt.value(2)
  // CHECK: constructed vt.value(1)
  // CHECK: move constructed vt.value(1)
  // CHECK: constructed vt.value(3)
  // CHECK: move constructed vt.value(3)
  // CHECK: print vt.value(3)
  %ch_out = grt.invoke @repeat_n_inc(%count) : (si32) -> !mef.chain
  mef.output %ch_out : !mef.chain
}
