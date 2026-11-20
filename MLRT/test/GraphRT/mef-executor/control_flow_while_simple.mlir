// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

// CHECK-LABEL: --- Running 'controlflow_while_simple'
// CHECK-NEXT: int32 = 0
// CHECK-NEXT: int32 = 0
// CHECK-NEXT: int32 = 1
// CHECK-NEXT: int32 = 1
// CHECK-NEXT: int32 = 2
// CHECK-NEXT: int32 = 2
// CHECK-NEXT: int32 = 3
// CHECK-NEXT: int32 = 3
// CHECK-NEXT: int32 = 4
// CHECK-NEXT: int32 = 4
// CHECK-NEXT: int32 = 5
// CHECK-NEXT: hello mef executor!
// CHECK-NEXT: int32 = 5
// CHECK-NEXT: int32 = 4
grt.testcase @controlflow_while_simple {
  %ch = grt.chain.create ()
  %i = grt.constant.si32 0
  %n = grt.constant.si32 4

  %res:3 = grt.while (%ch: !mef.chain, %i: si32, %n: si32) {
    %new_ch = grt.print.si32 %i, %ch
    %cond = grt.lessequal.si32 %i, %n
    mef.output %new_ch, %i, %n, %cond: !mef.chain, si32, si32, i1
  } do {
    %new_ch = grt.print.si32 %i, %ch
    %one = grt.constant.si32 1
    %new_i = grt.add.si32 %i, %one
    mef.output %new_ch, %new_i, %n: !mef.chain, si32, si32
  }

  %ch1 = "test.print_hello"(%res#0) : (!mef.chain) -> !mef.chain
  %ch2 = grt.print.si32 %res#1, %ch1
  %ch3 = grt.print.si32 %res#2, %ch2

  mef.output
}
