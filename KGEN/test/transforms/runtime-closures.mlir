// RUN: kgen-opt %s --runtime-closures -allow-unregistered-dialect | FileCheck %s

// CHECK: kgen.func @h(%arg0: index) -> index {
kgen.func @main_closure_arg(%arg0: index) {
  // CHECK: kgen.create_closure [(index) -> index: @h](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  } { name = "h" }
  kgen.return
}

// CHECK: kgen.func @two_captures(%arg0: index, %arg1: index, %arg2: index) -> index {
kgen.func @capturing_region(%arg0: index, %arg1: index) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure [(index, index, index) -> index: @two_captures](%arg0, %arg1)
  %0 = kgen.stage_closure = (%arg2: index) capturing -> index {
    "unregistered_op_to_capture"(%arg0, %arg1) : (index, index) -> ()
    kgen.return %arg2 : index
  } { name = "two_captures" }
  %1 = kgen.call_signature %0(%idx4) : (index) capturing -> index
  kgen.return
}

// CHECK:   kgen.func @stage_closure(%arg0: index) -> index {
// CHECK:   kgen.func @stage_closure_0(%arg0: index) -> index {
kgen.func @multiple_staged_closures_no_name_attr(%arg0: index, %arg1: index) {
  // CHECK: kgen.create_closure [(index) -> index: @stage_closure](%arg0)
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg0 : index
  }
  // CHECK: kgen.create_closure [(index) -> index: @stage_closure_0](%arg1)
  %1 = kgen.stage_closure = () capturing -> index {
    kgen.return %arg1 : index
  } { name = 6 }
  kgen.return
}

// CHECK: kgen.func @stage_closure_1() -> index {
kgen.func @constant_in(%arg0: index, %arg1: index) {
  %idx4 = index.constant 4
  // CHECK: kgen.create_closure [() -> index: @stage_closure_1]()
  %0 = kgen.stage_closure = () capturing -> index {
    kgen.return %idx4 : index
  }
  kgen.return
}

// -----

// COM: Ensure that captures are replaced only in stage closure region.
kgen.func @user(%arg0: !pop.pointer<struct<pointer<scalar<index>>, scalar<index>, scalar<index>>>) {
  kgen.return
}

// CHECK: kgen.func @nested_function(%arg0: !pop.pointer<struct<pointer<scalar<index>>, scalar<index>, scalar<index>>>) {
// CHECK: %0 = kgen.call @user(%arg0) : (!pop.pointer<struct<pointer<scalar<index>>, scalar<index>, scalar<index>>>) -> index
// CHECK: kgen.return
kgen.func @bind_nested_function(%arg0: index) {
  %4 = pop.stack_allocation 1 x !pop.struct<pointer<scalar<index>>, scalar<index>, scalar<index>>
  %W = kgen.call @user(%4) : (!pop.pointer<struct<pointer<scalar<index>>, scalar<index>, scalar<index>>>) -> index
  %9 = kgen.stage_closure = () capturing -> () {
    %11 = kgen.call @user(%4) : (!pop.pointer<struct<pointer<scalar<index>>, scalar<index>, scalar<index>>>) -> index
    kgen.return
  } {name = "nested_function"}
  kgen.return
}

kgen.func @main() {
   %idx39 = index.constant 39
   kgen.call @bind_nested_function(%idx39) : (index) -> ()
   kgen.return
}
