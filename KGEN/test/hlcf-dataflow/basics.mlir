// RUN: kgen-opt -test-dataflow %s 2>%t
// RUN: cat %t | FileCheck %s

kgen.func @simple_if(%cond: i1) -> (index, index) {
  %0 = hlcf.if %cond -> index {
    %1 = index.constant 0
    hlcf.yield %1 : index
  } else {
    %1 = index.constant 1
    hlcf.yield %1 : index
  }
  %2 = hlcf.if %cond -> index {
    %1 = index.constant 2
    hlcf.yield %1 : index
  } else {
    %1 = index.constant 2
    hlcf.yield %1 : index
  }

  // CHECK: simple_if(<UNKNOWN>, 2 : index)
  kgen.return {print_operand_constants = "simple_if"} %0, %2 : index, index
}

kgen.func @simple_loop() -> (index, index) {
  %0 = index.constant 0
  %1 = hlcf.loop () -> index {
    hlcf.break %0 : index
  }
  %2 = hlcf.loop (%arg0 = %0 : index) -> index {
    %3 = index.cmp eq(%arg0, %0)
    %4 = index.constant 1
    hlcf.if %3 {
      hlcf.break %4 : index
    } else {
      hlcf.yield
    }
    hlcf.continue %4 : index
  }

  // CHECK: simple_loop(0 : index, 1 : index)
  kgen.return {print_operand_constants = "simple_loop"} %1, %2 : index, index
}

kgen.func @multiple_returns(%cond: i1) -> index attributes {sym_visibility = "private"} {
  hlcf.if %cond {
    %0 = index.constant 0
    kgen.return %0 : index
  } else {
    %0 = index.constant 1
    kgen.return %0 : index
  }
  %0 = hlcf.loop () -> index {
    hlcf.continue
  }
  kgen.return %0 : index
}

kgen.func @call_multiple_returns() -> index {
  %0 = index.bool.constant true
  %1 = kgen.call @multiple_returns(%0) : (i1) -> index
  // CHECK: multiple_returns(0 : index)
  kgen.return {print_operand_constants = "multiple_returns"} %1 : index
}

kgen.func @unreachable_return() -> index {
  %0 = hlcf.loop () -> index {
    %0 = index.bool.constant false
    hlcf.if %0 {
      hlcf.yield
    } else {
      hlcf.continue
    }
    %1 = index.constant 0
    hlcf.break %1 : index
  }
  // CHECK: unreachable(<UNINITIALIZED>)
  kgen.return {print_operand_constants = "unreachable"} %0 : index
}

kgen.func @return_and_terminator(%cond: i1) -> index {
  %0 = hlcf.if %cond -> index {
    %c0 = index.constant 0
    kgen.return %c0 : index
  } else {
    %c1 = index.constant 1
    hlcf.yield %c1 : index
  }
  // CHECK: return_yield(1 : index)
  kgen.return {print_operand_constants = "return_yield"} %0 : index
}
