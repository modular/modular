// RUN: kgen-opt %s --runtime-closures -allow-unregistered-dialect | FileCheck %s

kgen.func @take_closure_no_args(%arg0: !kgen.signature<() capturing -> index>) {
    %0 = kgen.call_signature  %arg0() : () capturing -> index
    kgen.return
}

// CHECK: kgen.func @h(%arg0: index) -> index always_inline {
kgen.func @main_closure_arg(%arg0: index) {
    // CHECK: %0 = kgen.create_closure @h(%arg0) : (!kgen.signature<(index) -> index>, index) -> !kgen.signature<() capturing -> index>
    %0 = kgen.stage_closure = () capturing -> index {
      kgen.return %arg0 : index
    } { name = "h" }
    kgen.call @take_closure_no_args(%0) : (!kgen.signature<() capturing -> index>) -> ()
    kgen.return
}

// CHECK: kgen.func @two_captures(%arg0: index, %arg1: index, %arg2: index) -> index always_inline {
kgen.func @capturing_region(%arg0: index, %arg1: index) {
    %idx4 = index.constant 4
    // CHECK: %0 = kgen.create_closure @two_captures(%arg0, %arg1) : (!kgen.signature<(index, index, index) -> index>, index, index) -> !kgen.signature<(index) capturing -> index>
    %0 = kgen.stage_closure = (%arg2: index) capturing -> index {
        "unregistered_op_to_capture"(%arg0, %arg1) : (index, index) -> ()
        kgen.return %arg2 : index
    } { name = "two_captures" }
    %1 = kgen.call_signature %0(%idx4) : (index) capturing -> index
    kgen.return
}

// CHECK:   kgen.func @stage_closure(%arg0: index) -> index always_inline {
// CHECK:   kgen.func @stage_closure_0(%arg0: index) -> index always_inline {
kgen.func @multiple_staged_closures_no_name_attr(%arg0: index, %arg1: index) {
    // CHECK: %0 = kgen.create_closure @stage_closure(%arg0) : (!kgen.signature<(index) -> index>, index) -> !kgen.signature<() capturing -> index>
    %0 = kgen.stage_closure = () capturing -> index {
      kgen.return %arg0 : index
    }
    // CHECK: %1 = kgen.create_closure @stage_closure_0(%arg1) : (!kgen.signature<(index) -> index>, index) -> !kgen.signature<() capturing -> index>
    %1 = kgen.stage_closure = () capturing -> index {
      kgen.return %arg1 : index
    } { name = 6 }
    kgen.call @take_closure_no_args(%0) : (!kgen.signature<() capturing -> index>) -> ()
    kgen.call @take_closure_no_args(%1) : (!kgen.signature<() capturing -> index>) -> ()
    kgen.return
}

// CHECK: kgen.func @stage_closure_1() -> index always_inline {
kgen.func @constant_in(%arg0: index, %arg1: index) {
    %idx4 = index.constant 4
    // CHECK: %0 = kgen.create_closure @stage_closure_1() : (!kgen.signature<() -> index>) -> !kgen.signature<() capturing -> index>
    %0 = kgen.stage_closure = () capturing -> index {
      kgen.return %idx4 : index
    }
    kgen.call @take_closure_no_args(%0) : (!kgen.signature<() capturing -> index>) -> ()
    kgen.return
}
