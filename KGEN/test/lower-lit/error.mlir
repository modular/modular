// RUN: kgen-opt -lower-lit -allow-unregistered-dialect %s | FileCheck %s

lit.struct.decl @Error {}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !pop.variant<@Error, array<0, i1>>
lit.func @return_raise_or(%cond: i1, %err: !kgen.declref<@Error>) -> !pop.variant<@Error, !lit.none> {
  hlcf.if %cond {
    // CHECK: %[[ERR:.*]] = pop.variant.create %arg1
    %0 = pop.variant.create %err : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    // CHECK-NEXT: kgen.return %[[ERR]]
    kgen.return %0 : !pop.variant<@Error, !lit.none>
  } else {
    hlcf.yield
  }

  %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK: %[[VAL:.*]] = pop.variant.create %{{.*}}
  %1 = pop.variant.create %0 : !lit.none -> !pop.variant<@Error, !lit.none>
  // CHECK-NEXT: kgen.return %[[VAL]]
  kgen.return %1 : !pop.variant<@Error, !lit.none>
}

// CHECK-LABEL: kgen.generator @removeMetadata
// CHECK-SAME: (%arg0: !pop.pointer<index>) throws ->
lit.func @removeMetadata(%arg0: !pop.pointer<index> byref) throws -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}
