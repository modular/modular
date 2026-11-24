// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

grt.testcase @controlflow_repeat_large {
  %count = grt.constant.si32 1000
  %v0 = grt.constant.si32 42

  %sum = grt.repeat.i32 %count (%v0: si32) {
    %one = grt.constant.si32 1
    %v2 = "test.async.add.i32"(%v0, %one) : (si32, si32) -> si32
    mef.output %v2: si32
  }

  %ch0 = grt.chain.create ()
  // CHECK: int32 = 1042
  grt.print.si32 %sum, %ch0

  mef.output
}
