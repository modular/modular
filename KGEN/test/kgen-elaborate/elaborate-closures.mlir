// RUN: kgen-opt %s -elaborate-generators | FileCheck %s

kgen.generator @take_closure(%arg0: !kgen.signature<(index) capturing -> index>, %arg1: index) {
  %0 = kgen.call_signature %arg0(%arg1) : (index) capturing -> index
  kgen.return
}

// CHECK-LABEL: kgen.func @main_make_closure_no_param
kgen.generator @main_make_closure_no_param() {
  kgen.param.declare.region g = (%arg0: index) capturing {
    kgen.return
  }
  // CHECK: kgen.stage_closure = (%arg0: index) capturing {
  %0 = kgen.create_closure [<>(index) capturing -> (): g]()
  kgen.return
}

// CHECK-LABEL: kgen.func @main_make_closure_with_param
kgen.generator @main_make_closure_with_param() {
  kgen.param.declare.region h = <N>() capturing {
    %4 = kgen.param.constant = <N>
    kgen.return
  }
  // CHECK: [[ARG0:%[0-9]+]] = kgen.stage_closure = () capturing {
  // CHECK-NEXT: [[ARG1:%[0-9]+]] = kgen.param.constant = <3>
  // CHECK: [[ARG2:%[0-9]+]] = kgen.stage_closure = () capturing {
  // CHECK-NEXT: [[ARG3:%[0-9]+]] = kgen.param.constant = <4>
  kgen.param.declare Bound1: <>() capturing -> () = <bind_signature(:<index>() capturing -> () h, 3)>
  kgen.param.declare Bound2: <>() capturing -> () = <bind_signature(:<index>() capturing -> () h, 4)>
  %0 = kgen.create_closure [<>() capturing -> (): Bound1]()
  %3 = kgen.create_closure [<>() capturing -> (): Bound2]()
  kgen.return
}

// CHECK-LABEL: kgen.func @make_closure_param
kgen.generator @make_closure_param(%arg0: index) {
  kgen.param.declare.region g = <N>() capturing -> index {
      %6 = kgen.param.constant = <N>
      kgen.return %arg0 : index
  }
  // CHECK: kgen.stage_closure = () capturing -> index
  %0 = kgen.create_closure [<>() capturing -> index: bind_signature(:<index>() capturing -> index g, 4)]()
  kgen.return
}

// CHECK-LABEL: kgen.func @"make_closure_capture_param,N=5"
kgen.generator @make_closure_capture_param<N>(%arg0: index) {
  kgen.param.declare.region g = <M>() capturing -> index {
      %6 = kgen.param.constant = <N>
      kgen.return %arg0 : index
  }
  // CHECK: [[ARG0:%[0-9]+]] = kgen.stage_closure = () capturing -> index {
  // CHECK: [[ARG1:%[0-9]+]] = kgen.param.constant = <5>
  %0 = kgen.create_closure [<>() capturing -> index: bind_signature(:<index>() capturing -> index g, 4)]()
  kgen.return
}

// CHECK-LABEL: kgen.func @main_make_closure_capture_param
kgen.generator @main_make_closure_capture_param() {
  %idx4 = index.constant 4
  kgen.param.declare Bound: (index) -> () = <bind_signature(:<index>(index) -> () @make_closure_capture_param, 5)>
  // CHECK: kgen.call @"make_closure_capture_param,N=5"(%idx4) : (index) -> ()
  kgen.call_param[(index) -> (): Bound](%idx4)
  kgen.return
}

// CHECK-LABEL: kgen.func @main_make_closure_with_dtype_param
kgen.generator @main_make_closure_with_dtype_param() {
  %idx4 = index.constant 4

  kgen.param.declare.region k = <dt:dtype>() capturing -> index {
    kgen.return %idx4 : index
  }
  // CHECK: kgen.stage_closure = () capturing -> index {
  // CHECK: kgen.stage_closure = () capturing -> index {
  %0 = kgen.create_closure[<>() capturing -> index: bind_signature(:<dtype>() capturing -> index k, si32)]()
  %1 = kgen.create_closure[<>() capturing -> index: bind_signature(:<dtype>() capturing -> index k, f32)]()
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

// CHECK-LABEL: kgen.func @main_make_closure_with_symbol_param
kgen.generator @main_make_closure_with_symbol_param() {
  %idx4 = index.constant 4
  kgen.param.declare.region h = <fn: () -> index>() capturing -> index {
    %9 = kgen.call_param[() -> index: fn]()
    kgen.return %idx4 : index
  }
  // CHECK: kgen.stage_closure = () capturing -> index {
  // CHECK: kgen.call @foo_3() : () -> index
  // CHECK: kgen.stage_closure = () capturing -> index {
  // CHECK: kgen.call @foo_4() : () -> index
  kgen.param.declare Bound1: <>() capturing -> index = <bind_signature(:<() -> index>() capturing -> index h, @foo_3)>
  kgen.param.declare Bound2: <>() capturing -> index = <bind_signature(:<() -> index>() capturing -> index h, @foo_4)>
  %0 = kgen.create_closure[<>() capturing -> index: Bound1]()
  %1 = kgen.create_closure[<>() capturing -> index: Bound2]()
  kgen.return
}

// COM: Ensure that regions lifted by OutlineClosures pass are not erased
// CHECK-LABEL: kgen.func @"foo_k,N=5,M=3"() capturing -> !pop.scalar<index> {
kgen.generator @foo_k<N, M>() capturing -> !pop.scalar<index> {
  %0 = kgen.param.constant: scalar<index> = <0>
  kgen.return %0 : !pop.scalar<index>
}

// CHECK-LABEL: kgen.func @"foo,N=5"(%arg0: !pop.scalar<index>) {
kgen.generator @foo<N>(%arg0: !pop.scalar<index>) {
  kgen.param.declare k: <index>() capturing -> !pop.scalar<index> = <@foo_k<N, #kgen.unbound>>
  // CHECK: kgen.create_closure [<>() capturing -> !pop.scalar<index>: @"foo_k,N=5,M=3"]()
  %1 = kgen.create_closure[<>() capturing -> !pop.scalar<index>: bind_signature(:<index>() capturing -> !pop.scalar<index> k, 3)]()
  kgen.return
}

// CHECK-LABEL: kgen.func @main
kgen.generator @main() {
  %simd = kgen.param.constant: scalar<index> = <0>
  kgen.param.declare Bound: (!pop.scalar<index>) -> () = <@foo<5>>
  // CHECK: kgen.call @"foo,N=5"(%simd) : (!pop.scalar<index>) -> ()
  kgen.call_param[(!pop.scalar<index>) -> (): Bound](%simd)
  kgen.return
}

// COM: Ensure that staged closures follow the global store
kgen.generator @take_bat(%arg0: !kgen.signature<(index) capturing -> index>) {
	kgen.return
}

kgen.generator @bat(%arg0: index) capturing -> index {
	kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.func @bat_binder
kgen.generator @bat_binder(%arg0: index) {
  // CHECK: kgen.create_closure [<>(index) capturing -> index: @bat]()
	%2 = kgen.create_closure[<>(index) capturing -> index: h]()
	kgen.param.declare h: <>(index) capturing -> index = <@bat>
	kgen.return
}
