// RUN: mo-opt %s --mo-hoist-param-exprs | FileCheck %s

// CHECK-LABEL: mo.graph @complex_expression
mo.graph @complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  // CHECK-NEXT: mo.concat
  %1 = mo.concat[0](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  // CHECK-NEXT: kgen.param.declare [[NAME:[a-zA-Z0-9_]+]]: scalar<si64> = <add
  // CHECK-NEXT: mo.mul({{.*}}) : !mo.tensor<[[[NAME]]], f32>
  // CHECK-NEXT: rebind
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add({{.*}})], f32>
  %2 = mo.mul(%1, %1) :!mo.tensor<[add(N, M)], f32>
  mo.output %2 : !mo.tensor<[add(N, M)], f32>
}

// CHECK-LABEL: mo.graph @multiple_complex_expression
mo.graph @multiple_complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>) {
  // CHECK-NEXT: mo.concat
  // CHECK-NEXT: mo.static.broadcast_to
  // CHECK-NEXT: mo.static.reshape
  %1 = mo.concat[0](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  %2 = mo.static.broadcast_to(%arg1):!mo.tensor<[M], f32> ->!mo.tensor<[N, M], f32>
  %3 = mo.static.reshape(%2):!mo.tensor<[N, M], f32> ->!mo.tensor<[mul_no_wrap(N, M)], f32>
  // CHECK-DAG: kgen.param.declare [[NAME_ADD:[a-zA-Z0-9_]+]]: scalar<si64> = <add
  // CHECK-DAG: kgen.param.declare [[NAME_MUL:[a-zA-Z0-9_]+]]: scalar<si64> = <mul_no_wrap
  // CHECK-DAG: kgen.param.declare [[NAME_ADD_MUL:[a-zA-Z0-9_]+]]: scalar<si64> = <add(mul_no_wrap
  // CHECK-NEXT: mo.concat{{.*}} : (!mo.tensor<[[[NAME_ADD]]], f32>, !mo.tensor<[[[NAME_MUL]]], f32>) -> !mo.tensor<[[[NAME_ADD_MUL]]], f32>
  // CHECK-NEXT: mo.rebind
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add(mul_no_wrap{{.*}})], f32>
  %4 = mo.concat[0](%1, %3) :(!mo.tensor<[add(N,M)], f32>, !mo.tensor<[mul_no_wrap(N,M)], f32>) -> !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>
  mo.output %4 : !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>
}

// An mo.if whose result carries a composite dim: HoistParameterExpressions now
// aliases the dim to a single parameter on the mo.if result AND on both branch
// yields (the alias is declared in the enclosing region). Without matching all
// three, the region-branch verifier rejects the op (the else branch yields a
// value defined outside the mo.if). The final value is rebound back to the
// original expression for the graph output.
// CHECK-LABEL: mo.graph @scoped_multiple_complex_expression
// CHECK: mo.if({{.*}}) -> !mo.tensor<[param_expr_{{[0-9]+}}], f32> {
// CHECK: mo.yield {{.*}} : !mo.tensor<[param_expr_{{[0-9]+}}], f32>
// CHECK: } else {
// CHECK-NEXT: mo.yield {{.*}} : !mo.tensor<[param_expr_{{[0-9]+}}], f32>
// CHECK: }
// CHECK: mo.output{{.*}} : !mo.tensor<[add(mul_no_wrap{{.*}})], f32>
mo.graph @scoped_multiple_complex_expression<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>, %arg2: !mo.tensor<[1], bool>) -> (!mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>) {
  %1 = mo.concat[0](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  %2 = mo.static.broadcast_to(%arg1):!mo.tensor<[M], f32> ->!mo.tensor<[N, M], f32>
  %3 = mo.static.reshape(%2):!mo.tensor<[N, M], f32> ->!mo.tensor<[mul_no_wrap(N, M)], f32>
  %4 = mo.concat[0](%1, %3) :(!mo.tensor<[add(N,M)], f32>, !mo.tensor<[mul_no_wrap(N,M)], f32>) -> !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>
  %cond = mo.static.reshape(%arg2) : !mo.tensor<[1], bool> -> !mo.tensor<[], bool>
  %5 = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32> {
    %6 = mo.concat[0](%4) : (!mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>) -> !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>
    mo.yield %6 : !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>
  } else {
    mo.yield %4 : !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>
  }
  mo.output %5 : !mo.tensor<[add(add(N, M), mul_no_wrap(N,M))], f32>
}


// CHECK-LABEL: mo.graph @no_rewrite_decl
mo.graph @no_rewrite_decl<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  // CHECK-NEXT: mo.concat
  %1 = mo.concat[0](%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  %2 = mo.static.reshape(%1):!mo.tensor<[add(N, M)], f32>->!mo.tensor<[sum], f32>
  // CHECK-NEXT: kgen.param.declare [[SUM:.*]]: scalar<si64> = <add
  // CHECK-NEXT: mo.mul({{.*}}) : !mo.tensor<[[[SUM]]], f32>
  kgen.param.declare sum: scalar<si64> = <add(N, M)>
  %3 = mo.mul(%2, %2) :!mo.tensor<[sum], f32>
  // CHECK-NEXT: mo.rebind
  %4 = mo.static.reshape(%3):!mo.tensor<[sum], f32>->!mo.tensor<[add(N, M)], f32>
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add({{.*}})], f32>
  mo.output %4 : !mo.tensor<[add(N, M)], f32>
}

// CHECK-LABEL: mo.graph @rebind_output
// CHECK-SAME: [[D21:.*]], [[D22:.*]], [[D23:.*]], [[D24:.*]] -> [[D27:.*]]>([[VARG0:.*]]: !mo.tensor<[[[D21]], 12, [[D22]]], f32>, [[VARG1:.*]]: !mo.tensor<[[[D23]], [[D24]]], f32>) -> !mo.tensor<[[[D21]], 12, [[D27]]], f32> {
// [[V0:.*]] = mo.static.broadcast_to([[VARG0]]) : !mo.tensor<[[[D21]], 12, [[D22]]], f32> -> !mo.tensor<[[[D21]], 12, param_expr_1], f32>
// [[V1:.*]] = mo.static.broadcast_to([[VARG1]]) : !mo.tensor<[[[D23]], [[D24]]], f32> -> !mo.tensor<[[[D21]], 12, param_expr_1], f32>
// kgen.param.declare param_expr_1: scalar<si64> = <cond(eq([[D22]], 1), [[D24]], [[D22]])>
// [[V2:.*]] = mo.add([[V0]], [[V1]]) : !mo.tensor<[[[D21]], 12, param_expr_1], f32>
// [[V3:.*]] = mo.rebind([[V2]]) : !mo.tensor<[[[D21]], 12, param_expr_1], f32> -> !mo.tensor<[[[D21]], 12, cond(eq([[D22]], 1), [[D24]], [[D22]])], f32>
// mo.output<cond(eq([[D22]], 1), [[D24]], [[D22]])> [[V3]] : !mo.tensor<[[[D21]], 12, cond(eq([[D22]], 1), [[D24]], [[D22]])], f32>
mo.graph @rebind_output<D21, D22, D23, D24 -> D27>(%arg0: !mo.tensor<[D21, 12, D22], f32>, %arg1: !mo.tensor<[D23, D24], f32>) -> !mo.tensor<[D21, 12, D27], f32> {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[D21, 12, D22], f32> -> !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[D23, D24], f32> -> !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
  mo.output<cond(identical(:scalar<si64> D22, 1), D24, D22)> %2 : !mo.tensor<[D21, 12, cond(identical(:scalar<si64> D22, 1), D24, D22)], f32>
}

// COM: mo.rebind can't used with an opaque type.
// CHECK-LABEL: mo.graph @dont_rebind_opaque
// CHECK-SAME: <[[D0:[^,>]*]]>
mo.graph @dont_rebind_opaque<D0>(%arg0: !mo.tensor<[D0], f32>, %arg1: !mo.opaque<"Opaque">) -> (!mo.opaque<"Opaque">) {

  // CHECK-NEXT: kgen.param.declare [[D1:[^ ]*]]: scalar<si64> = <[[D0]]>
  // CHECK-NEXT: mo.rebind
  %0 = mo.rebind<() -> D1>(%arg0) : !mo.tensor<[D0], f32> -> !mo.tensor<[D1], f32>

  // CHECK-NEXT: mo.custom
  %1 = mo.custom [symbol = "custom0", device = #M.device_ref<"cpu", 0>](%0) : (!mo.tensor<[D1], f32>) -> !mo.tensor<[mul_no_wrap(D1, 2)], f32>

  // CHECK-NEXT: kgen.param.declare [[MUL:.*]]: scalar<si64> = <mul_no_wrap

  // CHECK-NEXT: (!mo.tensor<[[[MUL]]], f32>, !mo.opaque<"Opaque">) -> !mo.opaque<"Opaque">
  %2 = mo.custom [symbol = "custom", device = #M.device_ref<"cpu", 0>](%1, %arg1) : (!mo.tensor<[mul_no_wrap(D1, 2)], f32>, !mo.opaque<"Opaque">) -> !mo.opaque<"Opaque">

  // CHECK-NEXT: mo.output
  mo.output %2 : !mo.opaque<"Opaque">
}

// CHECK-LABEL: mo.graph @complex_expression_non_mo
mo.graph @complex_expression_non_mo<N, M>(%arg0: !mo.tensor<[N], f32>, %arg1: !mo.tensor<[M], f32>) -> (!mo.tensor<[add(N, M)], f32>) {
  // NOTE: this is a mogg.concat not a mo.concat
  // CHECK-NEXT: mogg.concat
  %1 = mogg.concat(%arg0, %arg1):(!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  // CHECK-NEXT: kgen.param.declare [[NAME:[a-zA-Z0-9_]+]]: scalar<si64> = <add
  // CHECK-NEXT: mo.mul({{.*}}) : !mo.tensor<[[[NAME]]], f32>
  // CHECK-NEXT: rebind
  // CHECK-NEXT: mo.output{{.*}} : !mo.tensor<[add({{.*}})], f32>
  %2 = mo.mul(%1, %1) :!mo.tensor<[add(N, M)], f32>
  mo.output %2 : !mo.tensor<[add(N, M)], f32>
}

// CHECK-LABEL: mo.graph @dont_outline_unknown_dims
mo.graph @dont_outline_unknown_dims<N, M>(%arg0: !mo.tensor<[N, ?], f32>, %arg1: !mo.tensor<[M, ?], f32>) -> (!mo.tensor<[add(N, M), ?], f32>) {
  // No question marks should disappear (there are 8 in the current IR)
  // CHECK-COUNT-8: ?
  // COUNT-NOT: ?
  %ax = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[], si64>
  %1 = mo.concat[1](%arg0, %arg1):(!mo.tensor<[N, ?], f32>, !mo.tensor<[M, ?], f32>) -> !mo.tensor<[add(N, M), ?], f32>
  %2 = mo.mul(%1, %1) :!mo.tensor<[add(N, M), ?], f32>
  mo.output %2 : !mo.tensor<[add(N, M), ?], f32>
}

// Regression test for GEX-3545: mo.call result types with parameter expressions
// must be rewritten to aliases. Previously, mo.call was skipped because it
// implements MOControlOpInterface, which left expressions in its result types
// and caused a "Missing dynamic dimension" assertion downstream in EmitMojo.
//
// mo.call is a MOControlOpInterface op, but its result types are plain
// attributes and are safe to rewrite (the output rebind restores the original
// type for downstream consumers, mirroring how mo.graph-returning ops are
// handled elsewhere in this pass).

// CHECK-LABEL: mo.graph @callee_subgraph
mo.graph @callee_subgraph<Aliased>(%arg0: !mo.tensor<[Aliased], f32>) -> !mo.tensor<[Aliased], f32> isSubgraph {
  mo.output %arg0 : !mo.tensor<[Aliased], f32>
}

// CHECK-LABEL: mo.graph @call_with_expr_result
mo.graph @call_with_expr_result<N, M, S>(%arg0: !mo.tensor<[S], f32>) -> !mo.tensor<[S], f32> {
  // The call must be rewritten to use an alias for the add(N, M) result dim.
  // CHECK-DAG: kgen.param.declare [[ALIAS:[a-zA-Z0-9_]+]]: scalar<si64> = <add(
  // CHECK: mo.call @callee_subgraph({{.*}}) {{.*}} -> !mo.tensor<[[[ALIAS]]], f32>
  %in = mo.rebind<() -> Aliased>(%arg0) {message = ""} : !mo.tensor<[S], f32> -> !mo.tensor<[Aliased], f32>
  %0 = mo.call @callee_subgraph(%in) : (!mo.tensor<[Aliased], f32>) -> !mo.tensor<[add(N, M)], f32>
  %out = mo.rebind(%0) {message = ""} : !mo.tensor<[add(N, M)], f32> -> !mo.tensor<[S], f32>
  mo.output %out : !mo.tensor<[S], f32>
}

// Regression test for mo.parallel with parameter-expression dims in
// `MO::BundleType` element types.

// CHECK-LABEL: mo.graph @parallel_bundle_param_expr
mo.graph @parallel_bundle_param_expr<D0>(
    %arg0: !mo.tensor<[D0, 3], f32>,
    %arg1: !mo.tensor<[D0, 3], f32>)
    -> (!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>) {
  // CHECK-DAG: kgen.param.declare [[ALIAS_PARALLEL:[a-zA-Z0-9_]+]]: scalar<si64> = <div(D0, 2)>
  %r0 = mo.rebind(%arg0) : !mo.tensor<[D0, 3], f32> -> !mo.tensor<[div(D0, 2), 3], f32>
  %r1 = mo.rebind(%arg1) : !mo.tensor<[D0, 3], f32> -> !mo.tensor<[div(D0, 2), 3], f32>
  // CHECK-DAG: mo.tensor.bundle{{.*}} -> !mo.bundle<[!mo.tensor<[[[ALIAS_PARALLEL]], 3], f32>, !mo.tensor<[[[ALIAS_PARALLEL]], 3], f32>]>
  %bundle = mo.tensor.bundle(%r0, %r1)
      : (!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>)
      -> (!mo.bundle<[!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>]>)
  // CHECK-DAG: mo.parallel {{.*}} : !mo.bundle<[!mo.tensor<[[[ALIAS_PARALLEL]], 3], f32>, !mo.tensor<[[[ALIAS_PARALLEL]], 3], f32>]>
  %res_bundle = mo.parallel (%x) in
      (%bundle : !mo.bundle<[!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>]>) {
    // CHECK-DAG: mo.relu({{.*}}) : !mo.tensor<[[[ALIAS_PARALLEL]], 3], f32>
    %1 = mo.relu(%x) : !mo.tensor<[div(D0, 2), 3], f32>
    mo.yield %1 : !mo.tensor<[div(D0, 2), 3], f32>
  }
  // CHECK-DAG: mo.tensor.unbundle{{.*}} -> (!mo.tensor<[[[ALIAS_PARALLEL]], 3], f32>, !mo.tensor<[[[ALIAS_PARALLEL]], 3], f32>)
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle)
      : (!mo.bundle<[!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>]>)
      -> (!mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>)
  mo.output %res_0, %res_1
      : !mo.tensor<[div(D0, 2), 3], f32>, !mo.tensor<[div(D0, 2), 3], f32>
}

// Regression test: mo.parallel whose operand bundle element types contain a
// parameter expression but whose result bundle does not (the body rebinds
// the algebraic-shaped block arg to a non-aliased dim before yielding).
//
// The op-level operand/result type checks on their own do not flag this
// parallel as containing a complex expression -- the bundle operand is a
// `MO::BundleType` (which doesn't set `containsComplexExprs`) and the
// result bundle's element type is `add(M, N)` only via the rebind, which
// the pass declares an alias for and rewrites in the bundle producer
// independently. Without also walking block-arg types, the parallel is
// skipped, the upstream `mo.tensor.bundle` producer is rewritten to use
// the alias for its result element types, and the parallel's block args
// remain in the original algebraic form -- producing a verifier mismatch
// against the rewritten bundle operand type on the next pattern
// application cycle.
//
// Fix: the matcher also walks the matched op's region block-arg types and
// nested op result types so the parallel is matched whenever any reachable
// type carries a hoistable expression.
//
// CHECK-LABEL: mo.graph @parallel_block_arg_param_expr
mo.graph @parallel_block_arg_param_expr<N, M>(
    %arg0: !mo.tensor<[N], f32>,
    %arg1: !mo.tensor<[M], f32>)
    -> (!mo.tensor<[N], f32>, !mo.tensor<[N], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[], si64>
  %sum = mo.concat[0](%arg0, %arg1)
      : (!mo.tensor<[N], f32>, !mo.tensor<[M], f32>) -> !mo.tensor<[add(N, M)], f32>
  // CHECK-DAG: kgen.param.declare [[ALIAS_BA:[a-zA-Z0-9_]+]]: scalar<si64> = <add
  // CHECK-DAG: mo.tensor.bundle{{.*}} -> !mo.bundle<[!mo.tensor<[[[ALIAS_BA]]], f32>, !mo.tensor<[[[ALIAS_BA]]], f32>]>
  %bundle = mo.tensor.bundle(%sum, %sum)
      : (!mo.tensor<[add(N, M)], f32>, !mo.tensor<[add(N, M)], f32>)
      -> (!mo.bundle<[!mo.tensor<[add(N, M)], f32>, !mo.tensor<[add(N, M)], f32>]>)
  // The parallel must be matched even though its result bundle element
  // type (`N`) contains no complex expression: the alias still has to
  // propagate into the body block args so they agree with the rewritten
  // bundle producer's output element types.
  // CHECK-DAG: mo.parallel ([[BA:%[a-zA-Z0-9_]+]]) in {{.*}} : !mo.bundle<[!mo.tensor<[[[ALIAS_BA]]], f32>, !mo.tensor<[[[ALIAS_BA]]], f32>]>
  // CHECK-DAG: mo.relu([[BA]]) : !mo.tensor<[[[ALIAS_BA]]], f32>
  %res_bundle = mo.parallel (%x) in
      (%bundle : !mo.bundle<[!mo.tensor<[add(N, M)], f32>, !mo.tensor<[add(N, M)], f32>]>)
      -> (!mo.bundle<[!mo.tensor<[N], f32>, !mo.tensor<[N], f32>]>) {
    %1 = mo.relu(%x) : !mo.tensor<[add(N, M)], f32>
    %2 = mo.rebind(%1) : !mo.tensor<[add(N, M)], f32> -> !mo.tensor<[N], f32>
    mo.yield %2 : !mo.tensor<[N], f32>
  }
  %r0, %r1 = mo.tensor.unbundle(%res_bundle)
      : (!mo.bundle<[!mo.tensor<[N], f32>, !mo.tensor<[N], f32>]>)
      -> (!mo.tensor<[N], f32>, !mo.tensor<[N], f32>)
  mo.output %r0, %r1 : !mo.tensor<[N], f32>, !mo.tensor<[N], f32>
}
