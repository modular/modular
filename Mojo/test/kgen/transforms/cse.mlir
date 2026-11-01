// RUN: kgen-opt -cse -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @cse_simple
kgen.func @cse_simple(%arg0: !kgen.scalar<bool>) -> index {
  // CHECK: [[INDEX:%.*]] = kgen.param.constant = <1>
  %index = kgen.param.constant: index = <1>

  // CHECK-NEXT: hlcf.if
  // CHECK: hlcf.yield [[INDEX]] : index
  // CHECK: hlcf.yield [[INDEX]] : index
  %0 = hlcf.if %arg0 -> index{
    %index_again = kgen.param.constant: index = <1>
    hlcf.yield %index_again : index
  } else {
    %index_again = kgen.param.constant: index = <1>
    hlcf.yield %index_again : index
  }
  // CHECK: hlcf.return [[INDEX]] : index
  hlcf.return %index : index
}
