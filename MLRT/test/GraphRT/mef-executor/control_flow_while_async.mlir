// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

// CHECK-LABEL: --- Running 'controlflow_while_async'
// CHECK-NEXT: int32 = 5
// CHECK-NEXT: int32 = 15
grt.testcase @controlflow_while_async {
  %ch = grt.chain.create ()
  %i = grt.constant.si32 0
  %sum = grt.constant.si32 0

  %res:2 = grt.while (%i: si32, %sum: si32) {
    %four = grt.constant.si32 4
    %cond = grt.lessequal.si32 %i, %four
    mef.output %i, %sum, %cond: si32, si32, i1
  } do {
    %one = grt.constant.si32 1
    %new_i = mef.opaque "test.async.add2.i32" nonstrict [](%i, %one) : (si32, si32) -> si32
    %new_sum = mef.opaque "test.async.add2.i32" nonstrict [](%sum, %new_i) : (si32, si32) -> si32
    mef.output %new_i, %new_sum: si32, si32
  }

  %ch1 = grt.print.si32 %res#0, %ch
  %ch2 = grt.print.si32 %res#1, %ch1

  mef.output
}
