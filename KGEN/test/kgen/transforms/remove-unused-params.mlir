
// RUN: kgen-opt --split-input-file --remove-unused-params --eliminate-dead-symbols %s  | FileCheck %s

kgen.generator @basic_arg_remove_1<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>,%arg2: !pop.scalar<T>) -> index{
  %l = pop.load %arg1 : !kgen.pointer<index>
  kgen.return %l : index
}

kgen.generator @basic_arg_remove_2<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) -> index{
  %0 = kgen.call @basic_arg_remove_1<:dtype T>(%arg0, %arg1,%arg2) : (index, !kgen.pointer<index>, !pop.scalar<T>) -> (index)
  kgen.return %0 : index
}

kgen.generator export @basic_arg_export<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) {
  %0 = kgen.call @basic_arg_remove_2<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !pop.scalar<T>) -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @basic_arg_export<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) {
// CHECK-NEXT: kgen.call @basic_arg_remove_2_REMOVED_ARG(%arg1) : (!kgen.pointer<index>) -> index

// CHECK-LABEL:  kgen.generator  @basic_arg_remove_1_REMOVED_ARG(%arg0: !kgen.pointer<index>) -> index
// CHECK-NEXT: %[[OUT:.*]] = pop.load %arg0 : !kgen.pointer<index>
// CHECK-NEXT: kgen.return %[[OUT]]

// CHECK-LABEL:  kgen.generator  @basic_arg_remove_2_REMOVED_ARG(%arg0: !kgen.pointer<index>) -> index
// CHECK-NEXT: %[[OUT:.*]] = kgen.call @basic_arg_remove_1_REMOVED_ARG(%arg0) : (!kgen.pointer<index>) -> index
// CHECK-NEXT: kgen.return %[[OUT]]


// -----

kgen.generator @recursive_test<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) -> index{
  %0 = kgen.call @recursive_test<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !pop.scalar<T>) -> (index)
  kgen.return %0 : index
}

kgen.generator export @recursive_test_entry<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) {
  %0 = kgen.call @recursive_test<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !pop.scalar<T>) -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @recursive_test_entry<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>)
// CHECK-NEXT: kgen.call @recursive_test_REMOVED_ARG() : () -> index
// CHECK-NEXT: kgen.return

// CHECK-LABEL:  kgen.generator  @recursive_test_REMOVED_ARG() -> index
// CHECK-NEXT: %[[OUT:.*]] = kgen.call @recursive_test_REMOVED_ARG() : () -> index
// CHECK-NEXT: kgen.return %[[OUT]]

// -----

// In this test X / Y are switched so only "Z" is unused.
// Technically we could add more logic to remove X / Y too but this is a tradeoff
// of code complexity as we need to guard against lots of cases, e.g <X, X + Y>

kgen.generator @recursive_test_2<X: index, Y: index, Z: index>() -> index {
  %0 = kgen.call @recursive_test_2<:index Y, :index X, :index Z>() : () -> (index)
  kgen.return %0 : index
}

kgen.generator export @recursive_test_2_entry<X: index, Y: index, Z: index>() {
  %0 = kgen.call @recursive_test_2<:index X, :index Y, :index Z>() : () -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @recursive_test_2_entry<X, Y, Z>()
// CHECK-NEXT: kgen.call @recursive_test_2_REMOVED_ARG<X, Y>() : () -> index
// CHECK-NEXT: kgen.return

// CHECK-LABEL:  kgen.generator  @recursive_test_2_REMOVED_ARG<X, Y>() -> index
// CHECK-NEXT: %[[OUT:.*]] = kgen.call @recursive_test_2_REMOVED_ARG<Y, X>() : () -> index
// CHECK-NEXT: kgen.return %[[OUT]]


// -----


kgen.generator @test_argument_captured_in_attr<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) attributes {some_metadata = [#kgen.param.decl.ref<"T"> : !kgen.dtype]} {
  kgen.return
}

kgen.generator export @test_argument_captured_in_attr_entry<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) {
  kgen.call @test_argument_captured_in_attr<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !pop.scalar<T>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @test_argument_captured_in_attr_entry<T: dtype>
// CHECK-NEXT: kgen.call @test_argument_captured_in_attr_REMOVED_ARG<:dtype T>() : () -> (

// CHECK-LABEL: kgen.generator  @test_argument_captured_in_attr_REMOVED_ARG<T: dtype>() attributes {some_metadata = [#kgen.param.decl.ref<"T"> : !kgen.dtype]}

// -----


kgen.generator @used_in_param_expr_test<T: dtype>(%arg0: index, %arg1: index) -> index{
  kgen.return %arg0 : index
}

kgen.generator export @used_in_param_expr_test_entry<type: dtype>(%arg0: index, %arg1: index) -> index {
  kgen.param.declare *"OUT`" = <apply(:(index, index) -> index @used_in_param_expr_test<:dtype f32>, 5, 10)>
  %0 = kgen.call @used_in_param_expr_test<:dtype f32>(%arg0, %arg1) : (index, index) -> index
  kgen.return %0 : index
}

// The old one has to exist for the parameter.
// CHECK-LABEL:kgen.generator  @used_in_param_expr_test<T: dtype>(%arg0: index, %arg1: index) -> index {
// CHECK-NEXT: kgen.return

// Param should use old one, call uses the one with the parameters removed.
// CHECK-LABEL:kgen.generator  export @used_in_param_expr_test_entry<type: dtype>(%arg0: index, %arg1: index) -> index {
// CHECK-NEXT: kgen.param.declare *"OUT`" = <apply(:(index, index) -> index @used_in_param_expr_test<:dtype f32>, 5, 10)>
// CHECK-NEXT: kgen.call @used_in_param_expr_test_REMOVED_ARG(%arg0) : (index) -> index

// CHECK-LABEL: kgen.generator  @used_in_param_expr_test_REMOVED_ARG(%arg0: index) -> index


// -----

kgen.generator @with_cycle_3(%arg0: index) -> index{
  %0 = kgen.call @with_cycle_2(%arg0) : (index) -> (index)
  kgen.return %0 : index
}

kgen.generator @with_cycle_2(%arg0: index) -> index{
  %0 = kgen.call @with_cycle_3(%arg0) : (index) -> (index)
  kgen.return %0 : index
}

kgen.generator @with_cycle_1<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) -> index{
  %0 = kgen.call @with_cycle_2(%arg0) : (index) -> (index)
  kgen.return %0 : index
}

kgen.generator export @with_cycle<T: dtype>(%arg0: index, %arg1: !kgen.pointer<index>, %arg2: !pop.scalar<T>) {
  %0 = kgen.call @with_cycle_1<:dtype T>(%arg0, %arg1, %arg2) : (index, !kgen.pointer<index>, !pop.scalar<T>) -> (index)
  kgen.return
}

// CHECK-LABEL: kgen.generator  export @with_cycle<
// CHECK-NEXT:  kgen.call @with_cycle_1_REMOVED_ARG

// CHECK-LABEL: kgen.generator  @with_cycle_1_REMOVED_ARG(%arg0: index) -> index {
// CHECK-NEXT: kgen.call @with_cycle_2(%arg0) : (index) -> index
