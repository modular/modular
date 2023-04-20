// RUN: kgen-opt %s -elaborate-generators | FileCheck %s

kgen.generator @take_closure(%arg0: !kgen.signature<(index) capturing -> index>, %arg1: index) {
  %0 = kgen.call_signature %arg0(%arg1) : (index) capturing -> index
  kgen.return
}

// CHECK: %0 = kgen.stage_closure = (%arg0: index) capturing -> index {
// CHECK: {name = "g_main_make_closure_no_param_concrete"}
kgen.generator @main_make_closure_no_param() {
  %idx4 = index.constant 4
  kgen.param.declare.region g = (%arg0: index) capturing -> index {
    kgen.return %idx4 : index
  }
  %0 = kgen.param.constant: <>(index) capturing -> index = <g>
  %idx3 = index.constant 3
  kgen.call @take_closure(%0, %idx3) : (!kgen.signature<(index) capturing -> index>, index) -> ()
  kgen.return
}

kgen.generator @take_closure_no_args(%arg0: !kgen.signature<() capturing -> index>) {
  %0 = kgen.call_signature %arg0() : () capturing -> index
  kgen.return
}

// CHECK: [[ARG0:%[0-9]+]] = kgen.stage_closure = () capturing -> index {
// CHECK: [[ARG1:%[0-9]+]] = kgen.param.constant = <3>
// CHECK: {name = "h_main_make_closure_with_param_concrete3"}
// CHECK: [[ARG2:%[0-9]+]] = kgen.stage_closure = () capturing -> index {
// CHECK: [[ARG3:%[0-9]+]] = kgen.param.constant = <4>
// CHECK: {name = "h_main_make_closure_with_param_concrete4"}
kgen.generator @main_make_closure_with_param() {
  %idx4 = index.constant 4
  kgen.param.declare.region h = <N>() capturing -> index {
    %4 = kgen.param.constant = <N>
    kgen.return %idx4 : index
  }
  kgen.param.declare Bound1: <>() capturing -> index = <bind_signature(:<index>() capturing -> index h, 3)>
  kgen.param.declare Bound2: <>() capturing -> index = <bind_signature(:<index>() capturing -> index h, 4)>
  %0 = kgen.param.constant: <>() capturing -> index = <Bound1>
  %3 = kgen.param.constant: <>() capturing -> index = <Bound2>
  kgen.call @take_closure_no_args(%0) : (!kgen.signature<() capturing -> index>) -> ()
  kgen.return
}

// CHECK: %0 = kgen.stage_closure = () capturing -> index {
// CHECK: {name = "g_make_closure_param_concrete4"}
kgen.generator @make_closure_param(%arg0: index) {
  kgen.param.declare.region g = <N>() capturing -> index {
      %6 = kgen.param.constant = <N>
      kgen.return %arg0 : index
  }
  %0 = kgen.param.constant: <>() capturing -> index = <bind_signature(:<index>() capturing -> index g, 4)>
  %1 = kgen.call_signature %0() : () capturing -> index
  kgen.return
}

// CHECK: [[ARG0:%[0-9]+]] = kgen.stage_closure = () capturing -> index {
// CHECK: [[ARG1:%[0-9]+]] = kgen.param.constant = <5>
// CHECK: {name = "g_make_closure_capture_param,N=54"}
kgen.generator @make_closure_capture_param<N>(%arg0: index) {
  kgen.param.declare.region g = <M>() capturing -> index {
      %6 = kgen.param.constant = <N>
      kgen.return %arg0 : index
  }
  %0 = kgen.param.constant: <>() capturing -> index = <bind_signature(:<index>() capturing -> index g, 4)>
  %1 = kgen.call_signature %0() : () capturing -> index
  kgen.return
}

kgen.generator @main_make_closure_capture_param() {
  %idx4 = index.constant 4
  kgen.param.declare Bound: (index) -> () = <bind_signature(:<index>(index) -> () @make_closure_capture_param, 5)>
  // CHECK: kgen.call @"make_closure_capture_param,N=5"(%idx4) : (index) -> ()
  kgen.call_param[(index) -> (): Bound](%idx4)
  kgen.return
}

// CHECK: %0 = kgen.stage_closure = () capturing -> index {
// CHECK: {name = "k_main_make_closure_with_dtype_param_concretesi32"}
// CHECK: %1 = kgen.stage_closure = () capturing -> index {
// CHECK: {name = "k_main_make_closure_with_dtype_param_concretef32"}
kgen.generator @main_make_closure_with_dtype_param() {
  %idx4 = index.constant 4

  kgen.param.declare.region k = <dt:dtype>() capturing -> index {
    kgen.return %idx4 : index
  }
  %0 = kgen.param.constant: <>() capturing -> index = <bind_signature(:<dtype>() capturing -> index k, si32)>
  %1 = kgen.param.constant: <>() capturing -> index = <bind_signature(:<dtype>() capturing -> index k, f32)>
  kgen.return
}

kgen.generator @foo_3() -> index {
  %idx3 = index.constant 3
  kgen.return %idx3 : index
}

kgen.generator @foo_4() -> index {
  %idx4 = index.constant 4
  kgen.return %idx4 : index
}

// CHECK: %0 = kgen.stage_closure = () capturing -> index {
// CHECK: %2 = kgen.call @foo_3() : () -> index
// CHECK: {name = "h_main_make_closure_with_symbol_param_concrete@foo_3!kgen.signature<() -> index>"}
// CHECK: %1 = kgen.stage_closure = () capturing -> index {
// CHECK: %2 = kgen.call @foo_4() : () -> index
// CHECK: {name = "h_main_make_closure_with_symbol_param_concrete@foo_4!kgen.signature<() -> index>"}
kgen.generator @main_make_closure_with_symbol_param() {
  %idx4 = index.constant 4
  kgen.param.declare.region h = <fn: () -> index>() capturing -> index {
    %9 = kgen.call_param[() -> index: fn]()
    kgen.return %idx4 : index
  }
  kgen.param.declare Bound1: <>() capturing -> index = <bind_signature(:<() -> index>() capturing -> index h, @foo_3)>
  kgen.param.declare Bound2: <>() capturing -> index = <bind_signature(:<() -> index>() capturing -> index h, @foo_4)>
  %0 = kgen.param.constant: <>() capturing -> index = <Bound1>
  %1 = kgen.param.constant: <>() capturing -> index = <Bound2>
  kgen.return
}
