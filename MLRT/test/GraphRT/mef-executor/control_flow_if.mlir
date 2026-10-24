// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

// This invokable is just a select: "cond ? v1 : v2", exercising grt.if with
// result values.
mef.callable @controlflow_if(%cond: i1, %v1: si32, %v2: si32) -> si32 {
  %res = grt.if %cond (%v1: si32, %v2: si32) -> (si32) {
    mef.output %v1 : si32
  } else {
    mef.output %v2 : si32
  }

  mef.output %res : si32
}

grt.testcase @if_test_0 {
  %ch0 = grt.chain.create ()

  %a = grt.constant.si32 45
  %b = grt.constant.si32 129

  %true = grt.constant.i1 1

  // CHECK: int32 = 45
  %x = grt.invoke @controlflow_if(%true, %a, %b) : (i1, si32, si32) -> (si32)
  %ch1 = grt.print.si32 %x, %ch0

  %false = grt.constant.i1 0

  // CHECK-NEXT: int32 = 129
  %y = grt.invoke @controlflow_if(%false, %a, %b) : (i1, si32, si32) -> (si32)
  %ch2 = grt.print.si32 %y, %ch1

  mef.output
}
