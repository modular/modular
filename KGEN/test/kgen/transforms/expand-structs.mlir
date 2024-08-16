// RUN: kgen-opt %s -expand-structs | FileCheck %s

// CHECK-LABEL: @struct_expand_if
kgen.func @struct_expand_if(%arg0: !kgen.struct<(i1, i2)>, %arg1: !kgen.struct<(i1, i2)>, %arg2: i1) -> !kgen.struct<(i1, i2)> {
  // CHECK: [[IN0:%.*]] = kgen.struct.create(%arg0, %arg1)
  // CHECK: [[IN1:%.*]] = kgen.struct.create(%arg2, %arg3)
  // CHECK: [[S:%.*]]:2 = hlcf.if %arg4 -> i1, i2
  %0 = hlcf.if %arg2 -> !kgen.struct<(i1, i2)> {
    // CHECK: [[R0:%.*]] = kgen.struct.extract [[IN0]][0]
    // CHECK-NEXT: [[R1:%.*]] = kgen.struct.extract [[IN0]][1]
    // CHECK-NEXT: yield [[R0]], [[R1]]
    hlcf.yield %arg0 : !kgen.struct<(i1, i2)>
  } else {
    // CHECK: [[R0:%.*]] = kgen.struct.extract [[IN1]][0]
    // CHECK-NEXT: [[R1:%.*]] = kgen.struct.extract [[IN1]][1]
    // CHECK-NEXT: yield [[R0]], [[R1]]
    hlcf.yield %arg1 : !kgen.struct<(i1, i2)>
  }
  // CHECK: kgen.struct.create([[S]]#0, [[S]]#1)
  kgen.return %0 : !kgen.struct<(i1, i2)>
}

// CHECK-LABEL: @struct_expand_loop
kgen.func @struct_expand_loop(%arg0: !kgen.struct<(i1, i2)>) -> !kgen.struct<(i1, i2)> {
  // CHECK: [[IN:%.*]] = kgen.struct.create(%arg0, %arg1)
  // CHECK-NEXT: [[R0:%.*]] = kgen.struct.extract [[IN]][0]
  // CHECK-NEXT: [[R1:%.*]] = kgen.struct.extract [[IN]][1]
  // CHECK-NEXT: [[S:%.*]]:2 = hlcf.loop (%arg2 = [[R0]] : i1, %arg3 = [[R1]] : i2) -> (i1, i2)
  %0 = hlcf.loop (%arg1 = %arg0 : !kgen.struct<(i1, i2)>) -> !kgen.struct<(i1, i2)> {
    // CHECK: [[ARG:%.*]] = kgen.struct.create(%arg2, %arg3)
    // CHECK-NEXT: [[R0:%.*]] = kgen.struct.extract [[ARG]][0]
    // CHECK-NEXT: [[R1:%.*]] = kgen.struct.extract [[ARG]][1]
    // CHECK-NEXT: continue [[R0]], [[R1]]
    hlcf.continue %arg1 : !kgen.struct<(i1, i2)>
  }
  // CHECK: kgen.struct.create([[S]]#0, [[S]]#1)
  kgen.return %0 : !kgen.struct<(i1, i2)>
}
