// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef --num-threads=1 | FileCheck %s
// RUN: mef-executor %t.mef --num-threads=4 | FileCheck %s

grt.testcase @test_async_add2 {
  %c1 = grt.constant.si32 1
  %c2 = grt.constant.si32 2
  %x = "test.async.add2.i32"(%c1, %c2) : (si32, si32) -> si32

  %y = "test.async.add2.i32"(%c1, %x) : (si32, si32) -> si32

  %z = "test.async.add2.i32"(%y, %y) : (si32, si32) -> si32

  // CHECK: 'test_async_add2' returned 3, 4, 8
  mef.output %x, %y, %z : si32, si32, si32
}
