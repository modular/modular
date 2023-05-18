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
