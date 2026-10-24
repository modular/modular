// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

grt.testcase @chain_merge {
  %ch0 = grt.chain.create ()
  // CHECK: int32 = 0
  %zero = grt.constant.si32 0
  %ch1 = grt.print.si32 %zero, %ch0

  // This merge doesn't add any synchronization: ch1 already depends on ch0.
  %ch2 = grt.chain.create (%ch0, %ch1)

  // These two prints aren't sequenced w.r.t. each other.

  // CHECK-DAG: int32 = 1
  %one = grt.constant.si32 1
  %ch3 = grt.print.si32 %one, %ch2

 // CHECK-DAG: int32 = 2
  %two = grt.constant.si32 2
  %ch4 = grt.print.si32 %two, %ch2

  // This merge waits for ch3/ch4 to complete.  ch1 isn't necessary here (but
  // doesn't hurt) because it is depended on through ch2 already.
  %ch5 = grt.chain.create (%ch3, %ch4, %ch1)

  // CHECK-NEXT: int32 = 0
  %ch6 = grt.print.si32 %zero, %ch5

  mef.output
}
