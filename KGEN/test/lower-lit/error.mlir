// RUN: kgen-opt -lower-lit -allow-unregistered-dialect %s | FileCheck %s

kgen.struct.decl @Error {}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !pop.variant<@Error, !kgen.list<i1[0]>>
lit.func @return_raise_or(%cond: i1, %err: !kgen.declref<@Error>) -> !pop.variant<@Error, !lit.none> {
  hlcf.if %cond {
    // CHECK: %[[ERR:.*]] = pop.variant.create %arg1
    %0 = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    // CHECK-NEXT: hlcf.return %[[ERR]]
    hlcf.return %0 : !pop.variant<@Error, !lit.none>
  } else {
    hlcf.yield
  }

  %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK: %[[VAL:.*]] = pop.variant.create %{{.*}}
  %1 = pop.variant.create %0 : !lit.none -> !pop.variant<@Error, !lit.none>
  // CHECK-NEXT: kgen.return %[[VAL]]
  kgen.return %1 : !pop.variant<@Error, !lit.none>
}

// CHECK-LABEL: kgen.generator.interface @removeConventions
// CHECK-SAME: (!pop.pointer<index>) throws ->
lit.func @removeConventions(%arg0: !pop.pointer<index> byref) throws -> !pop.variant<@Error, index> attributes {isInterface}
