// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

grt.testcase @controlflow_repeat2 {
  %count = grt.constant.si32 4
  %ch0 = grt.chain.create ()

  %v0 = grt.constant.si32 1

  // This loop properly passes around the control chain to guarantee proper
  // sequencing of the print calls.
  //
  // The body of a repeat is a invokable.  The "invoke" to it is passed the
  // %ch0 and %v0 values, the body of it receives them as %c1/%v1.
  %ch3:2 = grt.repeat.i32 %count (%ch0: !mef.chain, %v0: si32) {
    %ch2 = grt.print.si32 %v0, %ch0
    %v2 = grt.add.si32 %v0, %v0
    mef.output %ch2, %v2 : !mef.chain, si32
  }

  // CHECK: int32 = 1
  // CHECK-NEXT: int32 = 2
  // CHECK-NEXT: int32 = 4
  // CHECK-NEXT: int32 = 8

  %x = grt.constant.si32 -1

  // CHECK-NEXT: hello mef executor!
  %ch4 = "test.print_hello"(%ch3#0) : (!mef.chain) -> !mef.chain

  // CHECK-NEXT: int32 = 16
  grt.print.si32 %ch3#1, %ch4

  mef.output
}
