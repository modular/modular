// RUN: kgen-opt --slice-mogg-funcs %s | FileCheck %s

kgen.generator @fake_empty_tensor(%fake_pointer: !kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_tensor_allocator()")> {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

kgen.generator @fake_move_constructor(%fake_pointer: !kgen.pointer<struct<(index, index) memoryOnly>>, %fake_pointer2: !kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_tensor_move_constructor()")> {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}

kgen.generator @fake_user(%fake_pointer: !kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none {
    %none = kgen.param.constant: none = <#kgen.none>
    kgen.return %none : !kgen.none
}


kgen.generator @fake_kernel(%output: !kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none 
    decorators <:none apply(:() -> !kgen.none @"$utils::$_annotations::mogg_kgen_experiment_kernel()"), 
    :none apply(:(!kgen.string borrow, index borrow) -> !kgen.none @"$utils::$_annotations::mogg_register_override($builtin::$string_literal::StringLiteral,$builtin::$int::Int)", "mo.add", 1000)> {
    %none = kgen.param.constant: none = <#kgen.none>

    %0 = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
    %3 = kgen.call @fake_empty_tensor(%0) : (!kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none
    %4 = kgen.call @fake_user(%0) : (!kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none
    %5 = kgen.call @fake_move_constructor(%output, %0) : (!kgen.pointer<struct<(index, index) memoryOnly>>, !kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none

    kgen.return %none : !kgen.none
}


// Check the decorators have been stripped from the user kernel and
// that it's otherwise untouched.

// CHECK-LABEL: kgen.generator @fake_kernel(
// CHECK: %[[ARG0:.*]]: !kgen.pointer<struct<(index, index) memoryOnly>>) -> !kgen.none {
// CHECK: kgen.call @fake_empty_tensor
// CHECK: kgen.call @fake_user
// CHECK: kgen.call @fake_move_constructor

// Check the mogg facing sliced kernel has the decorators and has removed the
// internal allocation.

// CHECK: kgen.generator @fake_kernel_0(%[[OUTPUT:.*]]: !kgen.pointer<struct<(index, index) memoryOnly>>
// CHECK: decorators <
// CHECK-NOT: kgen.call @fake_empty_tensor
// CHECK: kgen.call @fake_user(%[[OUTPUT]])
// CHECK-NOT: kgen.call @fake_move_constructor
