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

// CHECK-LABEL: @unwrap_or_propagate
lit.func @unwrap_or_propagate(%cond: i1, %err: !kgen.declref<@Error>) -> !pop.variant<@Error, index> {
  // CHECK: %[[WRAPPED:.*]] = kgen.call @return_raise_or
  // CHECK-SAME: -> !pop.variant<@Error, !kgen.list<i1[0]>>
  %0 = kgen.call @return_raise_or(%cond, %err) : (i1, !kgen.declref<@Error>) -> !pop.variant<@Error, !lit.none>
  // CHECK-NEXT: %[[ISVAL:.*]] = pop.variant.is !kgen.list<i1[0]>, %[[WRAPPED]]
  // CHECK-NEXT: %[[VAL:.*]] = hlcf.if %[[ISVAL]]
  // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.list<i1[0]>
  // CHECK-NEXT:   hlcf.yield %[[UNWRAP]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.declref<@Error>
  // CHECK-NEXT:   %[[WRAPPED:.*]] = pop.variant.create %[[UNWRAP]] : !kgen.declref<@Error> -> !pop.variant<@Error, index>
  // CHECK-NEXT:   hlcf.return %[[WRAPPED]]
  %1 = lit.unwrap_or_propagate %0 : <@Error, !lit.none>
  // CHECK: "use"(%[[VAL]])
  "use"(%1) : (!lit.none) -> ()

  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK: %[[WRAPPED:.*]] = kgen.call @return_raise_or
    %2 = kgen.call @return_raise_or(%cond, %err) : (i1, !kgen.declref<@Error>) -> !pop.variant<@Error, !lit.none>
    // CHECK-NEXT: %[[ISVAL:.*]] = pop.variant.is !kgen.list<i1[0]>, %[[WRAPPED]]
    // CHECK-NEXT: %[[VAL:.*]] = hlcf.if %[[ISVAL]]
    // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.list<i1[0]>
    // CHECK-NEXT:   hlcf.yield %[[UNWRAP]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.declref<@Error>
    // CHECK-NEXT:   lit.try.raise %[[UNWRAP]]
    %3 = lit.unwrap_or_propagate %2 : <@Error, !lit.none>
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  }

  %2 = index.constant 0
  %3 = pop.variant.create %2 : index -> !pop.variant<@Error, index>
  kgen.return %3 : !pop.variant<@Error, index>
}

// CHECK-LABEL: kgen.generator.interface @removeConventions
// CHECK-SAME: (!pop.pointer<index>) ->
lit.func @removeConventions(%arg0: !pop.pointer<index> byref) throws -> !pop.variant<@Error, index> attributes {isInterface}
