// RUN: kgen-opt %s -elaborate-generators | FileCheck %s

!closure_wrapper_ptr = !kgen.pointer<struct<(pointer<none>, (!kgen.pointer<none> borrow, index borrow) -> index) memoryOnly>>

// CHECK-LABEL: kgen.func @"block_arg_needs_concretization
// CHECK-SAME: (%arg0: !kgen.pointer<struct<(index, !kgen.capture_list<(index borrow) capturing -> index : @"formatter2,TF=@formatter">) memoryOnly>> init_self) -> !kgen.none
kgen.generator @block_arg_needs_concretization<Z: (index borrow) capturing -> index>(%arg0: !kgen.pointer<struct<(index, !kgen.capture_list<(index borrow) capturing -> index : Z>) memoryOnly>> init_self) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.func @"noncallop_needs_concretization
kgen.generator @noncallop_needs_concretization<Z: (index borrow) capturing -> index>(%arg0: !closure_wrapper_ptr byref_result) capturing -> !kgen.none {

  // CHECK: [[W0:%.*]] = pop.stack_allocation 1 x struct<(index, !kgen.capture_list<(index borrow) capturing -> index : @"formatter2,TF=@formatter">) memoryOnly>
  %0 = pop.stack_allocation 1 x struct<(index, !kgen.capture_list<(index borrow) capturing -> index : Z>) memoryOnly>

  // CHECK-NEXT: kgen.call @"block_arg_needs_concretization,Z=@formatter2<:(index borrow) capturing -> index @formatter>"([[W0]]) : (!kgen.pointer<struct<(index, !kgen.capture_list<(index borrow) capturing -> index : @"formatter2,TF=@formatter">) memoryOnly>> init_self) -> !kgen.none
  %1 = kgen.call @"block_arg_needs_concretization"<:(index borrow) capturing -> index Z>(%0) : (!kgen.pointer<struct<(index, !kgen.capture_list<(index borrow) capturing -> index : Z>) memoryOnly>> init_self) capturing -> !kgen.none

  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @"formatter"(%arg0: index borrow) capturing -> index {
  kgen.return %arg0 : index
}

// CHECK: kgen.func @"formatter2,TF=@formatter"(%arg0: index borrow) capturing -> index
kgen.generator @"formatter2"<TF: (index borrow) capturing -> index>(%arg0: index borrow) capturing -> index {
  %2 = kgen.call_param[(index borrow) capturing -> index: TF](%arg0)
  kgen.return %2 : index
}

kgen.generator @main() {
  kgen.param.declare F: (index borrow) capturing -> index = <@"formatter">
  kgen.param.declare F2: (index borrow) capturing -> index = <@"formatter2"<:(index borrow) capturing -> index F>>

  %20 = pop.stack_allocation 1 x struct<(pointer<none>, (!kgen.pointer<none> borrow, index borrow) -> index) memoryOnly>
  %21 = kgen.call @noncallop_needs_concretization<:(index borrow) capturing -> index F2>(%20) : (!closure_wrapper_ptr byref_result) capturing -> !kgen.none
  kgen.return
}
