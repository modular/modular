// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

grt.testcase @controlflow_repeat5 {
  %ch0 = grt.chain.create ()
  %count = grt.constant.si32 5
  %ch1 = grt.repeat.i32 %count (%ch0: !mef.chain) {
    %ch1 = "test.print_hello"(%ch0) : (!mef.chain) -> !mef.chain

    %x = grt.constant.si32 0
    %ch2 = grt.print.si32 %x, %ch1

    mef.output %ch2 : !mef.chain
  }

  // CHECK: hello mef executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello mef executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello mef executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello mef executor!
  // CHECK-NEXT: int32 = 0
  // CHECK-NEXT: hello mef executor!
  // CHECK-NEXT: int32 = 0

  %x = grt.constant.si32 -1

  // CHECK-NEXT: int32 = -1
  grt.print.si32 %x, %ch1

  mef.output
}
