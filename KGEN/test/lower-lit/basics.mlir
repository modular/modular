// RUN: kgen-opt -lower-lit %s | FileCheck %s

// CHECK-LABEL: kgen.generator @trivial_generator
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32 {
// CHECK-NEXT:    kgen.return %[[ARG0]] : si32
// CHECK-NEXT:  }
lit.func @trivial_generator(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: kgen.generator @varDecl
// CHECK-SAME:  (%[[ARG0:.*]]: index) -> index {
// CHECK-NEXT:    %[[VAR_A:.*]] = pop.stack_allocation 1 x index
// CHECK-NEXT:    kgen.return %[[ARG0]] : index
// CHECK-NEXT:  }

lit.func @varDecl(%arg0: index) -> index {
  %a = lit.varlet.decl "a", var = true, synth=false : <index>
  kgen.return %arg0 : index
}

// CHECK-LABEL: kgen.generator @letDecl(%arg0
// CHECK-NEXT:    kgen.return %arg0 : index
lit.func @letDecl(%arg0: index) -> index {
  %a = lit.letreg.decl "a" = %arg0 : index
  %b = lit.letreg.decl "b" = %a : index
  kgen.return %b : index
}

//===----------------------------------------------------------------------===//
// Aliases
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @aliasFwdDecl()
// CHECK-NEXT: kgen.return
lit.func @aliasFwdDecl() {
  lit.alias.fwd.decl "xyz" : index
  kgen.return
}

//===----------------------------------------------------------------------===//
// HandleVariant
//===----------------------------------------------------------------------===//

lit.func @throwing_caller() throws -> !pop.variant<@Error, !lit.none> {
  %y = lit.varlet.decl "y", var = false, synth = false : <@MyStruct>
  %0 = kgen.call @throwing_callee(%y) : (!pop.pointer<@MyStruct> byref_result) throws -> !pop.variant<@Error, !lit.none>
  // CHECK: %2 = pop.variant.is !pop.array<0, i1>, %1 : !pop.variant<@Error, array<0, i1>>
  // CHECK: %3 = hlcf.if %2 -> !pop.array<0, i1> {
  // CHECK:   %4 = pop.variant.get %1 : !pop.variant<@Error, array<0, i1>> as !pop.array<0, i1>
  // CHECK:   hlcf.yield %4 : !pop.array<0, i1>
  // CHECK: } else {
  // CHECK:   %4 = pop.variant.get %1 : !pop.variant<@Error, array<0, i1>> as !kgen.declref<@Error>
  // CHECK:   %5 = pop.variant.create %4 : !kgen.declref<@Error> -> !pop.variant<@Error, array<0, i1>>
  // CHECK:   kgen.return %5 : !pop.variant<@Error, array<0, i1>>
  // CHECK:  }
  %1 = lit.handle_variant %0, %y : (!pop.variant<@Error, !lit.none>, !pop.pointer<@MyStruct>) -> !lit.none {
    %7 = pop.variant.get %0 : !pop.variant<@Error, !lit.none> as !lit.none
    lit.yield %7 : !lit.none
  } else {
    %8 = pop.variant.get %0 : !pop.variant<@Error, !lit.none> as !kgen.declref<@Error>
    %9 = pop.variant.create %8 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
    kgen.return %9 : !pop.variant<@Error, !lit.none>
  }
  %6 = kgen.param.constant: !pop.variant<@Error, !lit.none> = <#pop.variant<:!lit.none #lit.none>>
  kgen.return %6 : !pop.variant<@Error, !lit.none>
}

lit.func @caller_reg() -> !lit.none {
  lit.try {
    %0 = kgen.call @throwing_callee() : () throws -> !pop.variant<@Error, index>
    // CHECK: %[[VAR0:.*]] = pop.variant.is index, %0 : !pop.variant<@Error, index>
    // CHECK: %[[VAR1:.*]] = hlcf.if %[[VAR0]] -> index {
    // CHECK:   %[[VAR2:.*]] = pop.variant.get %0 : !pop.variant<@Error, index> as index
    // CHECK:   hlcf.yield %[[VAR2]] : index
    // CHECK: } else {
    // CHECK:   %[[VAR3:.*]] = pop.variant.get %0 : !pop.variant<@Error, index> as !kgen.declref<@Error>
    // CHECK:   lit.raise %[[VAR3]] : <@Error>
    // CHECK:   kgen.unreachable
    // CHECK: }
    %1 = lit.handle_variant %0 : (!pop.variant<@Error, index>) -> index {
      %7 = pop.variant.get %0 : !pop.variant<@Error, index> as index
      lit.yield %7 : index
    } else {
      %8 = pop.variant.get %0 : !pop.variant<@Error, index> as !kgen.declref<@Error>
      lit.raise %8 : !kgen.declref<@Error>
      kgen.unreachable
    }
    lit.try.yield
  } except (%arg0: !kgen.declref<@Error>) {
    lit.try.yield
  } else {
    lit.try.yield
  }
  %6 = kgen.param.constant: !lit.none = <#lit.none>
  kgen.return %6 : !lit.none
}

//===----------------------------------------------------------------------===//
// ErrorReturn
//===----------------------------------------------------------------------===//

lit.struct.decl @Error {}

lit.func @throwing_func() throws -> !pop.variant<@Error, !lit.none> {
  %1 = lit.struct.create() : () -> !kgen.declref<@Error>
  %2 = pop.variant.create %1 : !kgen.declref<@Error> -> !pop.variant<@Error, !lit.none>
  // CHECK: kgen.return %1 : !pop.variant<@Error, array<0, i1>>
  lit.error_return %2 : !pop.variant<@Error, !lit.none>
}
