// RUN: kgen-opt -lower-lit -allow-unregistered-dialect %s | FileCheck %s

kgen.struct.decl @Error {}

// CHECK-LABEL: kgen.generator @return_raise_or
// CHECK-SAME: -> !pop.variant<@Error, !kgen.list<i1[0]>>
lit.func @return_raise_or(%cond: i1, %err: !kgen.declref<@Error>) -> !lit.raises_or<!lit.none> {
  hlcf.if %cond {
    // CHECK: %[[ERR:.*]] = pop.variant.create %arg1 : !kgen.declref<@Error> -> !pop.variant<@Error, !kgen.list<i1[0]>>
    %0 = lit.raise_error %err : <@Error> -> <!lit.none>
    // CHECK-NEXT: hlcf.return %[[ERR]]
    hlcf.return %0 : !lit.raises_or<!lit.none>
  } else {
    hlcf.yield
  }

  %0 = kgen.param.constant: !lit.none = <#lit.none>
  // CHECK: %[[VAL:.*]] = pop.variant.create %{{.*}} : !kgen.list<i1[0]> -> !pop.variant<@Error, !kgen.list<i1[0]>>
  %1 = lit.form_value %0 : <!lit.none>
  // CHECK-NEXT: kgen.return %[[VAL]]
  kgen.return %1 : !lit.raises_or<!lit.none>
}

// CHECK-LABEL: @unwrap_or_propagate
lit.func @unwrap_or_propagate(%cond: i1, %err: !kgen.declref<@Error>) -> !lit.raises_or<index> {
  // CHECK: %[[WRAPPED:.*]] = kgen.call @return_raise_or
  // CHECK-SAME: -> !pop.variant<@Error, !kgen.list<i1[0]>>
  %0 = kgen.call @return_raise_or(%cond, %err) : (i1, !kgen.declref<@Error>) -> !lit.raises_or<!lit.none>
  // CHECK-NEXT: %[[ISVAL:.*]] = pop.variant.is !kgen.list<i1[0]>, %[[WRAPPED]]
  // CHECK-NEXT: %[[VAL:.*]] = hlcf.if %[[ISVAL]]
  // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.list<i1[0]>
  // CHECK-NEXT:   hlcf.yield %[[UNWRAP]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.declref<@Error>
  // CHECK-NEXT:   %[[WRAPPED:.*]] = pop.variant.create %[[UNWRAP]] : !kgen.declref<@Error> -> !pop.variant<@Error, index>
  // CHECK-NEXT:   hlcf.return %[[WRAPPED]]
  %1 = lit.unwrap_or_propagate %0 : <!lit.none>
  // CHECK: "use"(%[[VAL]])
  "use"(%1) : (!lit.none) -> ()

  // CHECK-NEXT: lit.try
  lit.try {
    // CHECK: %[[WRAPPED:.*]] = kgen.call @return_raise_or
    %2 = kgen.call @return_raise_or(%cond, %err) : (i1, !kgen.declref<@Error>) -> !lit.raises_or<!lit.none>
    // CHECK-NEXT: %[[ISVAL:.*]] = pop.variant.is !kgen.list<i1[0]>, %[[WRAPPED]]
    // CHECK-NEXT: %[[VAL:.*]] = hlcf.if %[[ISVAL]]
    // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.list<i1[0]>
    // CHECK-NEXT:   hlcf.yield %[[UNWRAP]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   %[[UNWRAP:.*]] = pop.variant.get %[[WRAPPED]] : !pop.variant<@Error, !kgen.list<i1[0]>> as !kgen.declref<@Error>
    // CHECK-NEXT:   lit.try.raise %[[UNWRAP]]
    %3 = lit.unwrap_or_propagate %2 : <!lit.none>
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  }

  %2 = index.constant 0
  %3 = lit.form_value %2 : <index>
  kgen.return %3 : !lit.raises_or<index>
}

// CHECK-LABEL: kgen.generator.interface @removeConventions
// CHECK-SAME: (!pop.pointer<index>) ->
lit.func @removeConventions(%arg0: !pop.pointer<index> byref) throws -> !lit.raises_or<index> attributes {isInterface}
