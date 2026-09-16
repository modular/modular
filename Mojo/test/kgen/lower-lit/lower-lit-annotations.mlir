// RUN: kgen-opt -verify-parameters -lower-lit -split-input-file %s | FileCheck %s

// `@__annotation` values ride along with the struct as it lowers, and each
// value's type is captured beside it. Lowering flattens a value to its storage
// form -- a single-field register-passable struct becomes its element type --
// so the type has to be recorded as a type value while it still names the
// nominal type.

// CHECK-LABEL: kgen.struct.generator @StructLevel
// CHECK-SAME:  annotationTypes = #kgen<exprs[#kgen.type<index> : !kgen.type, #kgen.type<string> : !kgen.type]>
// CHECK-SAME:  annotations = #kgen<exprs[7 : index, "tag" : !kgen.string]>
lit.struct.decl @StructLevel register_passable attributes {
  annotations = #kgen<exprs[7 : index, "tag" : !kgen.string]>
} {
  lit.struct.field value : index
}

// -----

// A struct with no annotations gets no annotation attributes at all, rather
// than empty ones.

// CHECK-LABEL: kgen.struct.generator @Plain
// CHECK-NOT: annotations
// CHECK-NOT: annotationTypes
lit.struct.decl @Plain register_passable {
  lit.struct.field value : index
}

// -----

// Field annotations are parallel with the fields, so an unannotated field
// holds an empty list rather than nothing: a null element would make the
// attribute walker and its replacement disagree on the element count.

// CHECK-LABEL: kgen.struct.generator @FieldLevel
// CHECK-SAME:  fieldAnnotationTypes = [#kgen<exprs[#kgen.type<index> : !kgen.type]>, #kgen<exprs[]>]
// CHECK-SAME:  fieldAnnotations = [#kgen<exprs[1 : index]>, #kgen<exprs[]>]
lit.struct.decl @FieldLevel {
  lit.struct.field first {annotations = #kgen<exprs[1 : index]>} : index
  lit.struct.field second : i32
}

// -----

// Annotations are written in the struct's own scope, so they can name its
// parameters. They lower unevaluated, to be rebound per instantiation.

// CHECK-LABEL: kgen.struct.generator @Parametric
// CHECK-SAME:  annotations = #kgen<exprs[#kgen.param.decl.ref<"N"> : index]>
lit.struct.decl @Parametric<N, T: type> register_passable attributes {
  annotations = #kgen<exprs[#kgen.param.decl.ref<"N"> : index]>
} {
  lit.struct.field value : index
}
