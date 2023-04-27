// RUN: kgen-opt %s --runtime-closures -allow-unregistered-dialect | FileCheck %s

// CHECK: kgen.func @h(%arg0: index) capturing -> index {
kgen.func @main_closure_arg(%arg0: index) {
  // CHECK: kgen.create_closure [<>(index) capturing -> index: @h](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  } { name = "h" }
  kgen.return
}

// CHECK: kgen.func @two_captures(%arg0: index, %arg1: index, %arg2: index) capturing -> index {
kgen.func @capturing_region(%arg0: index, %arg1: index) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure [<>(index, index, index) capturing -> index: @two_captures](%arg0, %arg1)
  %0 = kgen.stage_closure = (%arg2: index) capturing -> index {
    "unregistered_op_to_capture"(%arg0, %arg1) : (index, index) -> ()
    kgen.return %arg2 : index
  } { name = "two_captures" }
  %1 = kgen.call_signature %0(%idx4) : (index) capturing -> index
  kgen.return
}

// CHECK:   kgen.func @stage_closure(%arg0: index) capturing -> index {
// CHECK:   kgen.func @stage_closure_0(%arg0: index) capturing -> index {
kgen.func @multiple_staged_closures_no_name_attr(%arg0: index, %arg1: index) {
  // CHECK: kgen.create_closure [<>(index) capturing -> index: @stage_closure](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  }
  // CHECK: kgen.create_closure [<>(index) capturing -> index: @stage_closure_0](%arg1)
  %1 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg1 : index
  } { name = 6 }
  kgen.return
}

// CHECK: kgen.func @stage_closure_1() capturing -> index {
kgen.func @constant_in(%arg0: index, %arg1: index) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure [<>() capturing -> index: @stage_closure_1]()
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %idx4 : index
  }
  kgen.return
}
