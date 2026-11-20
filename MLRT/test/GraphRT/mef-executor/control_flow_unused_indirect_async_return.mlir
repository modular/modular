// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

// MEFExecutor will allocate an IndirectAsyncValue for this invokable's return
// value.
mef.callable @indirect_async_return(%c1: si32) -> si32 {
  %v2 = "test.async.add.i32"(%c1, %c1) : (si32, si32) -> si32
  %v3 = "test.async.add.i32"(%v2, %v2) : (si32, si32) -> si32
  // We can't return %v2 because it's an unavailable ConcreteAsyncValue (not an
  // IndirectAsyncValue).
  mef.output %v3 : si32
}

// Test an unused IndirectAsync return value.
grt.testcase @unused_indirect_async_return {
  %c1 = grt.constant.si32 1
  // The returned IndirectAsyncValue is unused, but it has not yet been
  // forwarded. Setting this IndirectAsyncValue must count as a use, otherwise
  // the executor would immediately destroy the unused value.
  grt.invoke @indirect_async_return(%c1) : (si32) -> si32

  // CHECK: Running
  mef.output
}
