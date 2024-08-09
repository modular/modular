// RUN: kgen-opt %s --mogg-outline | FileCheck %s

kgen.generator @call_target<_INPUT_FN_0: variant<() capturing -> !kgen.none, i1>>() capturing -> !kgen.none no_inline {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator export @BASE_KERNEL() capturing -> !kgen.none attributes {_in_lambdas = ["_INPUT_FN_0", "_INPUT_FN_1"], mogg.sliced} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.param.declare.region _INPUT_FN_1 = () capturing -> !kgen.none always_inline {
    %none1 = kgen.param.constant: none = <#kgen.none>
    kgen.return %none1 : !kgen.none
  }
  kgen.param.declare.region _INPUT_FN_0 = () capturing -> !kgen.none always_inline {
    %none2 = kgen.param.constant: none = <#kgen.none>
    kgen.return %none2 : !kgen.none
  }
  kgen.param.declare.region NON_INPUT_LAMBDA = () capturing -> !kgen.none always_inline {
    %none3 = kgen.param.constant: none = <#kgen.none>
    kgen.return %none3 : !kgen.none
  }
  %0 = kgen.call @call_target<:variant<() capturing -> !kgen.none, i1> #kgen.variant<:() capturing -> !kgen.none _INPUT_FN_0, 0>>() : () capturing -> !kgen.none
  %1 = kgen.call @call_target<:variant<() capturing -> !kgen.none, i1> #kgen.variant<:() capturing -> !kgen.none _INPUT_FN_1, 0>>() : () capturing -> !kgen.none
  %3 = pop.stack_allocation 1 x struct<(index, index)>
  kgen.return %none : !kgen.none
}

// Check we have outlined the code.

// CHECK-LABEL: kgen.generator @BASE_KERNEL_OUTLINED
// CHECK-SAME: <[[PARAM1:.*]]: () capturing -> !kgen.none, [[PARAM2:.*]]: () capturing -> !kgen.none>() capturing
// CHECK-SAME: no_inline
// CHECK-SAME: mogg.outlined

// CHECK-NEXT: kgen.param.constant: none
// CHECK-NEXT: kgen.param.declare.region NON_INPUT_LAMBDA
// CHECK-NEXT: kgen.param.constant
// CHECK-NEXT: kgen.return
// CHECK-NEXT:}
// CHECK-NEXT: kgen.call @call_target
// CHECK-NEXT: kgen.call @call_target
// CHECK-NEXT: pop.stack_allocation
// CHECK-NEXT: kgen.return

// Check we correctly call the outlined function.
// CHECK-LABEL: kgen.generator export @BASE_KERNEL

// Should have preserved the input lambdas.
// CHECK: kgen.param.declare.region _INPUT_FN_1
// CHECK: kgen.param.declare.region _INPUT_FN_0
// CHECK-NEXT: kgen.param.constant
// CHECK-NEXT: kgen.return
// CHECK-NEXT: }
// CHECK-NEXT: kgen.call @BASE_KERNEL_OUTLINED
// CHECK-NEXT: kgen.return


// Dummy for elementwise for each function.
kgen.generator @elementwise_gen<ty: dtype, ELEMWISE_BODY: () capturing -> !kgen.none>() capturing -> !kgen.none no_inline attributes {"mogg.elem_hook"} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}


kgen.generator export @KERNEL_WITH_ELEMENTWISE() capturing -> !kgen.none attributes {_in_lambdas = ["_INPUT_FN_0", "_INPUT_FN_1"], mogg.sliced} {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.param.declare.region _INPUT_FN_1 = () capturing -> !kgen.none always_inline {
    %none1 = kgen.param.constant: none = <#kgen.none>
    kgen.return %none1 : !kgen.none
  }
  kgen.param.declare.region _INPUT_FN_0 = () capturing -> !kgen.none always_inline {
    %none2 = kgen.param.constant: none = <#kgen.none>
    kgen.return %none2 : !kgen.none
  }
  kgen.param.declare.region _ELEMWISE_BODY = () capturing -> !kgen.none always_inline {
    kgen.param.declare.region NON_INPUT_LAMBDA = () capturing -> !kgen.none always_inline {
        %none3 = kgen.param.constant: none = <#kgen.none>
        kgen.return %none3 : !kgen.none
    }
    %0 = kgen.call @call_target<:variant<() capturing -> !kgen.none, i1> #kgen.variant<:() capturing -> !kgen.none _INPUT_FN_0, 0>>() : () capturing -> !kgen.none
    %1 = kgen.call @call_target<:variant<() capturing -> !kgen.none, i1> #kgen.variant<:() capturing -> !kgen.none _INPUT_FN_1, 0>>() : () capturing -> !kgen.none
    %3 = pop.stack_allocation 1 x struct<(index, index)>
    kgen.return %none : !kgen.none
  }

  %0 = kgen.call @elementwise_gen<:dtype f32, :() capturing -> !kgen.none _ELEMWISE_BODY>() : () capturing -> !kgen.none
  kgen.return %none : !kgen.none
}

// Check that for elementwise we correctly only outline the stuff in the
// elementwise lambda.
// CHECK-LABEL: kgen.generator export @KERNEL_WITH_ELEMENTWISE

// Should have preserved the input lambdas.
// CHECK: kgen.param.declare.region _INPUT_FN_1
// CHECK: kgen.param.declare.region _INPUT_FN_0
// CHECK-NEXT: kgen.param.constant
// CHECK-NEXT: kgen.return
// CHECK-NEXT: }
// CHECK-NEXT: kgen.param.declare.region _ELEMWISE_BODY
// CHECK-NEXT: kgen.call @KERNEL_WITH_ELEMENTWISE_OUTLINED
// CHECK-NEXT: kgen.return
// CHECK-NEXT: }
// CHECK-NEXT: kgen.call @elementwise_gen
// CHECK-NEXT: kgen.return
