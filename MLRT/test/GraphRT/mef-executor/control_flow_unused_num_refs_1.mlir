// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

grt.testcase @print_num_refs_1 {
  %ch0 = grt.chain.create ()
  %0 = grt.constant.si32 0
  %1 = grt.debug.print_num_refs %0 : si32
  // CHECK: operand has 1 refs
  // CHECK-NEXT: int32 = 0
  %ch1 = grt.print.si32 %1, %ch0
  mef.output
}
