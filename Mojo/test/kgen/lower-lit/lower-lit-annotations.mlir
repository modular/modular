// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file %s | FileCheck %s

// `@__annotation` values ride along with the struct as it lowers. Each one is a
// `#kgen.annotation` that pairs the value with its type value, and both halves
// are lowered.

// CHECK-LABEL: kgen.struct.generator @StructLevel
// CHECK-SAME:  annotations = [#kgen.annotation<7 : index, index>, #kgen.annotation<"tag" : !kgen.string, string>]
lit.struct.decl @StructLevel register_passable attributes {
  annotations = [#kgen.annotation<7 : index, index>, #kgen.annotation<"tag" : !kgen.string, !kgen.string>]
} {
  lit.struct.field value : index
}

// -----

// A struct with no annotations gets an empty list.

// CHECK-LABEL: kgen.struct.generator @Plain
// CHECK-SAME:  (value: index)>
// CHECK-SAME:  annotations = []
lit.struct.decl @Plain register_passable {
  lit.struct.field value : index
}

// -----

// Field annotations are carried on the field in the struct instance type. An
// unannotated field prints no annotation list.

// CHECK-LABEL: kgen.struct.generator @FieldLevel
// CHECK-SAME:  (first: index[#kgen.annotation<1 : index, index>], second: i32)
lit.struct.decl @FieldLevel {
  lit.struct.field first {annotations = [#kgen.annotation<1 : index, index>]} : index
  lit.struct.field second : i32
}

// -----

// Annotations are written in the struct's own scope, so they can name its
// parameters. They lower unevaluated, to be rebound per instantiation.

// CHECK-LABEL: kgen.struct.generator @Parametric
// CHECK-SAME:  annotations = [#kgen.annotation<#kgen.param.decl.ref<"N"> : index, index>]
lit.struct.decl @Parametric<N: index, T: type> register_passable attributes {
  annotations = [#kgen.annotation<#kgen.param.decl.ref<"N"> : index, index>]
} {
  lit.struct.field value : index
}
