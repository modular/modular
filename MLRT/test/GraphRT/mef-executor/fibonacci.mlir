// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef
// RUN: mef-executor %t.mef | FileCheck %s

mef.callable @fib(%arg: si32) -> si32 {
  %one = grt.constant.si32 1
  %cond = grt.lessequal.si32 %arg, %one
  %res = grt.if %cond (%arg: si32) -> si32 {
    mef.output %arg : si32
  } else {
    %min_one = grt.constant.si32 -1
    %n_min_one = grt.add.si32 %arg, %min_one
    %n_min_two = grt.add.si32 %n_min_one, %min_one
    %fib_n_min_one = grt.invoke @fib(%n_min_one) : (si32) -> (si32)
    %fib_n_min_two = grt.invoke @fib(%n_min_two) : (si32) -> (si32)
    %sum = grt.add.si32 %fib_n_min_one, %fib_n_min_two
    mef.output %sum : si32
  }

  mef.output %res : si32
}

// CHECK-LABEL: --- Running 'fib_driver'
grt.testcase @fib_driver {
  %zero = grt.constant.si32 0
  %fib0 = grt.invoke @fib(%zero) : (si32) -> (si32)

  %four = grt.constant.si32 4
  %fib4 = grt.invoke @fib(%four) : (si32) -> (si32)

  %eight = grt.constant.si32 8
  %fib8 = grt.invoke @fib(%eight) : (si32) -> (si32)

  %sixteen = grt.constant.si32 16
  %fib16 = grt.invoke @fib(%sixteen) : (si32) -> (si32)

  %ch0 = grt.chain.create ()

  // CHECK: int32 = 0
  %ch1 = grt.print.si32 %fib0, %ch0

  // CHECK: int32 = 3
  %ch2 = grt.print.si32 %fib4, %ch1

  // CHECK: int32 = 21
  %ch3 = grt.print.si32 %fib8, %ch2

  // CHECK: int32 = 987
  %ch4 = grt.print.si32 %fib16, %ch3

  mef.output
}
