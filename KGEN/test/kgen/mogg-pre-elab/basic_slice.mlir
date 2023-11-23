// RUN: kgen-opt --slice-mogg-funcs %s | FileCheck %s

kgen.generator @fake_empty_tensor(%fake_pointer: !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_tensor_allocator()")> {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

kgen.generator @fake_move_constructor(%fake_pointer: !kgen.pointer<struct<() memoryOnly>>, %fake_pointer2: !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_tensor_move_constructor()")> {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

kgen.generator @fake_user(%fake_pointer: !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}


kgen.generator @fake_kernel(%output: !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none 
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_kgen_experiment_kernel()"), 
    :none apply(:(!kgen.string borrow, index borrow) -> !kgen.none @"$utils::$_annotations::mogg_register_override($builtin::$string_literal::StringLiteral,$builtin::$int::Int)", "mo.add", 1000)> {
    %none = kgen.param.constant: none = <#kgen.none>

    %0 = pop.stack_allocation 1 x struct<() memoryOnly>
    %3 = kgen.call @fake_empty_tensor(%0) : (!kgen.pointer<struct<() memoryOnly>>) -> !kgen.none
    %4 = kgen.call @fake_user(%0) : (!kgen.pointer<struct<() memoryOnly>>) -> !kgen.none
    %5 = kgen.call @fake_move_constructor(%output, %0) : (!kgen.pointer<struct<() memoryOnly>>, !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none

    kgen.return %none : !kgen.none
}


// Check the decorators have been stripped from the user kernel and
// that it's otherwise untouched.

// CHECK-LABEL: kgen.generator @fake_kernel(
// CHECK: %[[ARG0:.*]]: !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none {
// CHECK: kgen.call @fake_empty_tensor
// CHECK: kgen.call @fake_user
// CHECK: kgen.call @fake_move_constructor

// Check the mogg facing sliced kernel has the decorators and has removed the
// internal allocation.

// CHECK: kgen.generator @fake_kernel_0(%[[OUTPUT:.*]]: !kgen.pointer<struct<() memoryOnly>>
// CHECK: decorators <
// CHECK-NOT: kgen.call @fake_empty_tensor
// CHECK: kgen.call @fake_user(%[[OUTPUT]])
// CHECK-NOT: kgen.call @fake_move_constructor

// A representative call that enables fusion on this tensor.
kgen.generator @fake_enable_fusion(%fake_pointer: !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_enable_fusion()")> {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}



kgen.generator @fake_elementwise<_4x5_x, _5x5_y, _6x5_z, _7x5_input_lambda: () capturing -> index, _8x5_output_lambda: () capturing -> index, _19x20_param: () capturing -> index>(%arg0: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) capturing -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_elementwise_hook()")> {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

// This is the call we look at to see how mojo binds the parameter.
kgen.generator @sample_call<_4x5_x, _5x5_y, _6x5_z, _7x5_input_lambda: () capturing -> index, _8x5_output_lambda: () capturing -> index>() capturing -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// This contains the canonical output lambda.
kgen.generator @fake_input_fusion_hook<_4x5_x, _5x5_y, _6x5_z, _7x5_input_lambda: () capturing -> index, _8x5_output_lambda: () capturing -> index>(%arg0: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) capturing -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_input_fusion_hook()")> {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.param.declare.region *"test()" = () capturing -> index {
    %index = kgen.param.constant = <_4x5_x>
    kgen.return %index : index
  }
  %0 = kgen.call @sample_call<_4x5_x, _5x5_y, _6x5_z, :() capturing -> index *"test()", :() capturing -> index _8x5_output_lambda>() : () capturing -> !kgen.none
  kgen.return %none : !kgen.none
}


kgen.generator @fake_user_with_params<_4x5_x, _5x5_y, _6x5_z, _7x5_input_lambda: () capturing -> index, _8x5_output_lambda: () capturing -> index>() capturing -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @fake_kernel_input_fusion<_21x8_x0, _21x8_x1, _21x8_x2, _21x8_x3: () capturing -> index, _21x8_x4: () capturing -> index>(%output: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem, %input: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_kgen_experiment_kernel()"), 
    :none apply(:(!kgen.string borrow, index borrow) -> !kgen.none @"$utils::$_annotations::mogg_register_override($builtin::$string_literal::StringLiteral,$builtin::$int::Int)", "mo.add", 1000)> {
    %none = kgen.param.constant: none = <#kgen.none>

    %0 = pop.stack_allocation 1 x struct<() memoryOnly>
    %3 = kgen.call @fake_empty_tensor(%0) : (!kgen.pointer<struct<() memoryOnly>>) -> !kgen.none
    %5 = kgen.call @fake_move_constructor(%output, %0) : (!kgen.pointer<struct<() memoryOnly>>, !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none


    %b = kgen.call @fake_enable_fusion<_21x8_x0, _21x8_x1, _21x8_x2, :() capturing -> index _21x8_x3, :() capturing -> index _21x8_x4>(%input) : (!kgen.pointer<struct<() memoryOnly>> borrow_in_mem) capturing -> !kgen.none

    %use = kgen.call @fake_user_with_params<_21x8_x0, _21x8_x1, _21x8_x2, :() capturing -> index _21x8_x3, :() capturing -> index _21x8_x4>(%input) : (!kgen.pointer<struct<() memoryOnly>> borrow_in_mem) capturing -> !kgen.none

    kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @fake_kernel_input_fusion_1

// Check that the output lambda has been annotated on the signature.
// CHECK: (%[[OUTPUT:.*]]: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem, %[[INPUT:.*]]: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none attributes {_in_lambdas = ["input_0_fn"], _out_lambdas = [""]}

// Check the lambda has been pulled into this function and the parameter has
// been remapped to point to "this" tensor parameter 
// CHECK: kgen.param.declare.region input_0_fn
// CHECK-NEXT %index = kgen.param.constant = <_21x8_x0>


// Check the function the input tensor is passed to now accepts this 
// output function.
// CHECK: kgen.call @fake_user_with_params<_21x8_x0, _21x8_x1, _21x8_x2, :() capturing -> index input_0_fn, :() capturing -> index _21x8_x4>(%[[INPUT]])



kgen.generator @fake_kernel_elementwise_fusion<_21x8_x0, _21x8_x1, _21x8_x2, _21x8_x3: () capturing -> index, _21x8_x4: () capturing -> index>(%output: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem, %input: !kgen.pointer<struct<() memoryOnly>> borrow_in_mem) -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_kgen_experiment_kernel()"), 
    :none apply(:(!kgen.string borrow, index borrow) -> !kgen.none @"$utils::$_annotations::mogg_register_override($builtin::$string_literal::StringLiteral,$builtin::$int::Int)", "mo.add", 1000)> {
    %none = kgen.param.constant: none = <#kgen.none>

    %0 = pop.stack_allocation 1 x struct<() memoryOnly>
    %3 = kgen.call @fake_empty_tensor(%0) : (!kgen.pointer<struct<() memoryOnly>>) -> !kgen.none
    %5 = kgen.call @fake_move_constructor(%output, %0) : (!kgen.pointer<struct<() memoryOnly>>, !kgen.pointer<struct<() memoryOnly>>) -> !kgen.none

    kgen.param.declare.region *"my_func()" = () capturing -> index {
      %index42 = kgen.param.constant = <42>
      kgen.return %index42 : index
    } {isolated}

    %o = kgen.call @fake_elementwise<_21x8_x0, _21x8_x1, _21x8_x2, :() capturing -> index _21x8_x3, :() capturing -> index _21x8_x4, :() capturing -> index *"my_func()">(%input) : (!kgen.pointer<struct<() memoryOnly>> borrow_in_mem) capturing -> !kgen.none

    kgen.return %none : !kgen.none
}

// No input or output fusions and check the elementwise function has been marked.
// CHECK-LABEL: kgen.generator @fake_kernel_elementwise_fusion_
// CHECK: {_elementwise_lambda = "my_func()", _in_lambdas = [""], _out_lambdas = [""]}
