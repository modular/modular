// RUN: kgen-opt -sccp -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @loop_generates_constant
kgen.func @loop_generates_constant() -> (index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK: [[FALSE:%.*]] = index.bool.constant false
  // CHECK-DAG: [[IDX11:%.*]] = index.constant 11
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // The result of this loop will be 2
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    // CHECK: index.cmp
    %3 = index.cmp slt(%arg0, %idx2)
    hlcf.if %3 {
      hlcf.yield
    } else {
      %4 = index.add %arg0, %1
      // CHECK: hlcf.break [[IDX11]]
      hlcf.break %4: index
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5 : index
  }

  // %2 will be a constant, so this cmp result will be a constant
  // CHECK-NOT: index.cmp
  %6 = index.cmp slt(%2, %idx2)

  // CHECK: [[V1:%.*]] = hlcf.if [[FALSE]]
  %7 = hlcf.if %6 -> index {
    hlcf.yield %idx0: index
  } else {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // CHECK: kgen.return [[IDX11]], [[IDX9]]
  kgen.return %2, %7 : index, index
}

// CHECK-LABEL: @not_much_can_be_known
kgen.func @not_much_can_be_known(%cond: i1) -> (index, index) {
  // Not much can be folded except obvious one that has constant operands.
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0
  %2 = hlcf.loop(%arg0 = %idx0: index, %arg1 = %cond: i1) -> index {
    %3 = hlcf.if %arg1 -> index {
      hlcf.yield %idx0: index
    } else {
      hlcf.yield %idx1: index
    }

    %4 = index.cmp slt(%3, %arg0)
    hlcf.if %4 {
      hlcf.yield
    } else {
      %5 = index.add %3, %3
      hlcf.break %5: index
    }

    %6 = index.add %3, %idx1
    %7 = index.cmp slt(%3, %idx2)
    hlcf.continue %6, %7 : index, i1
  }

  // CHECK: kgen.return [[IDX9]], [[V0:%.*]]
  kgen.return %1, %2 : index, index
}

// CHECK-LABEL: @nested_if_constant_result
kgen.func @nested_if_constant_result(%cond: i1) -> index {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK: [[TRUE:%.*]] = index.bool.constant true
  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  %2 = hlcf.if %cond -> index {
    %3:2 = hlcf.if %cond -> index, index {
      // CHECK: hlcf.yield [[IDX9]], [[IDX1]]
      hlcf.yield %1, %idx1: index, index
    } else {
      // CHECK: hlcf.yield [[IDX2]], [[IDX1]]
      hlcf.yield %idx2, %idx1: index, index
    }
    kgen.call @foo(%3#0) : (index) -> ()
    // This cmp has constant result.
    %4 = index.cmp slt (%3#1, %idx2)

    // CHECK: [[V2:%.*]] = hlcf.if [[TRUE]]
    %5 = hlcf.if %4 -> index {
      // CHECK: hlcf.yield [[IDX1]]
      hlcf.yield %3#1: index
    } else {
      hlcf.yield %3#0: index
    }

    // CHECK: hlcf.yield [[IDX1]]
    hlcf.yield %5: index
  } else {
    // CHECK: hlcf.yield [[IDX3]]
    hlcf.yield %0: index
  }

  kgen.return %2 : index
}


// CHECK-LABEL: @loop_generates_constant_but_hits_limit
kgen.func @loop_generates_constant_but_hits_limit() -> (index, index) {
  %idx0 = index.constant 0
  %idx1 = index.constant 1
  %idx2 = index.constant 2
  %idx110 = index.constant 110

  %0 = index.add %idx1, %idx2
  %1 = index.mul %0, %0

  // CHECK-DAG: [[IDX9:%.*]] = index.constant 9
  // CHECK-DAG: [[IDX3:%.*]] = index.constant 3
  // CHECK-DAG: [[IDX110:%.*]] = index.constant 110
  // CHECK-DAG: [[IDX2:%.*]] = index.constant 2
  // CHECK-DAG: [[IDX1:%.*]] = index.constant 1
  // CHECK-DAG: [[IDX0:%.*]] = index.constant 0

  // The result of this loop will be 110, but hits analysis threshold before finishing,
  // so result will be unknown.
  // CHECK: [[V2:%.*]] = hlcf.loop
  %2 = hlcf.loop(%arg0 = %idx0: index) -> index {
    // CHECK: index.cmp
    %3 = index.cmp slt(%arg0, %idx110)
    hlcf.if %3 {
      hlcf.yield
    } else {
      %4 = index.add %arg0, %1
      hlcf.break %4: index
    }
    %5 = index.add %arg0, %idx1
    hlcf.continue %5 : index
  }

  // CHECK: [[V6:%.*]] = index.cmp slt([[V2]], [[IDX2]])
  %6 = index.cmp slt(%2, %idx2)

  // CHECK: [[V7:%.*]] = hlcf.if [[V6:%.*]]
  %7 = hlcf.if %6 -> index {
    hlcf.yield %idx0: index
  } else {
    // CHECK: hlcf.yield [[IDX9]]
    hlcf.yield %1: index
  }

  // CHECK: kgen.return [[V2]], [[V7]]
  kgen.return %2, %7 : index, index
}
