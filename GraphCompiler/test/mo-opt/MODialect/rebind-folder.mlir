// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @rebind_param_materialization<[[A:.*]], [[B:.*]] ->  [[N:.*]],  [[M:.*]]>(
mo.graph @rebind_param_materialization<A, B -> N, M>(
  // CHECK-SAME: %arg0: !mo.tensor<[[[A]], [[B]]], f32>
  %arg0 : !mo.tensor<[A, B], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[N]], [[M]]], f32> {
) -> !mo.tensor<[N, M], f32> {
  // CHECK-DAG: kgen.param.declare [[K:.*]]: scalar<si64> = <[[A]]>
  // CHECK-DAG: kgen.param.declare [[L:.*]]: scalar<si64> = <[[B]]>
  // CHECK-DAG: %[[VAL0:.*]] = mo.rebind(%arg0) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[K]], [[L]]], f32>
  %0 = mo.rebind<() -> K, L>(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[K, L], f32>
  // CHECK: mo.output<[[K]], [[L]]> %[[VAL0]] : !mo.tensor<[[[K]], [[L]]], f32>
  mo.output<K, L> %0 : !mo.tensor<[K, L], f32>
}


// Key is only one definition of param C
// CHECK: mo.graph @rebind_reuse_declared_param<A, B -> O>
// CHECK-NEXT: kgen.param.declare C: scalar<si64> = <A>
// CHECK-NOT: kgen.param.declare
// CHECK-NEXT: mo.rebind({{.*}}) : !mo.tensor<[A, B], f32> -> !mo.tensor<[C, C], f32>
mo.graph @rebind_reuse_declared_param<A, B -> O>(
  %arg0: !mo.tensor<[A, B], f32>
) -> !mo.tensor<[O, O], f32> {
  %0 = mo.rebind<() -> C>(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[C, C], f32>
  mo.output<C> %0 : !mo.tensor<[C, C], f32>
}
