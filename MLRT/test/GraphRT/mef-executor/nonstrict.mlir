// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef | FileCheck %s

mef.callable @return_first_arg(%x: si32, %y: si32) -> si32 {
  mef.output %x : si32
}

// Test a nonstrict call with an unused argument.
// CHECK-LABEL: --- Running 'unused_nonstrict_arg'
grt.testcase @unused_nonstrict_arg {
  %c1 = grt.constant.si32 1

  %v2 = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %v3 = "test.async.add.i32"(%v2, %v2) : (si32, si32) -> si32

  // mefExecutor will allocate an IndirectAsyncValue for %v3, but %v3 is unused.
  %x = grt.invoke @return_first_arg(%c1, %v3) : (si32, si32) -> si32

  // At this point %v3 is unused, but has not yet been forwarded. Setting %v3
  // must count as a use, otherwise the executor would immediately destroy %v3.

  %ch0 = grt.chain.create ()
  // CHECK-NEXT: int32 = 1
  %ch1 = grt.print.si32 %x, %ch0
  mef.output
}

// CHECK-LABEL: --- Running 'many_waiters'
grt.testcase @many_waiters {
  %ch0 = grt.chain.create ()

  %v3 = "test.async.many.waiters"() { num = 3 : si32 } : () -> si32
  %ch1 = grt.print.si32 %v3, %ch0
  // CHECK-NEXT: int32 = 3

  %v6 = "test.async.many.waiters"() { num = 6 : si32 } : () -> si32
  %ch2 = grt.print.si32 %v6, %ch1
  // CHECK-NEXT: int32 = 6

  %v42 = "test.async.many.waiters"() { num = 42 : si32 } : () -> si32
  %ch3 = grt.print.si32 %v42, %ch2
  // CHECK-NEXT: int32 = 42
  mef.output
}

// CHECK-LABEL: --- Running 'test_many_tasks'
grt.testcase @test_many_tasks {
  %ch0 = grt.chain.create ()

  %v3 = "test.many.tasks"() { num = 3 : si32 } : () -> si32
  %ch1 = grt.print.si32 %v3, %ch0
  // CHECK-NEXT: int32 = 3

  %v129 = "test.many.tasks"() { num = 129 : si32 } : () -> si32
  %ch2 = grt.print.si32 %v129, %ch1
  // CHECK-NEXT: int32 = 129

  %v1248 = "test.many.tasks"() { num = 1248 : si32 } : () -> si32
  %ch3 = grt.print.si32 %v1248, %ch2
  // CHECK-NEXT: int32 = 1248
  mef.output
}
