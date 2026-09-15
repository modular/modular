// RUN: kgen-opt %s -split-input-file -verify-parameters -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -split-input-file -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// `@__annotation` values attached to a struct generator, read back with
// `struct_annotation_types` (the whole list, as types) and `struct_annotation`
// (one value by index). A negative field index selects the struct's own
// annotations, otherwise the ones on that field.
//
// `struct_annotation`'s type is derived rather than written: it is the
// `index`th element of `struct_annotation_types` for the same struct and
// field. Spelling that derivation out is what the verbose result types below
// are; Mojo writes them through a `comptime` alias, whose type is inferred.

kgen.struct.generator @Annotated
    = struct_inst<"Annotated"(first: index, second: i32)> attributes {
  annotationTypes = #kgen<exprs[#kgen.type<index> : !kgen.type,
                                #kgen.type<string> : !kgen.type]>,
  annotations = #kgen<exprs[7 : index, "tag" : !kgen.string]>,
  fieldAnnotationTypes = [#kgen<exprs[#kgen.type<index> : !kgen.type]>,
                          #kgen<exprs[]>],
  fieldAnnotations = [#kgen<exprs[1 : index]>, #kgen<exprs[]>]
}

#annotated = #kgen.type<typevalue<:type #kgen.genref<@Annotated>>,
                        struct<(index, i32)>> : !kgen.type

// CHECK-LABEL: kgen.func @read_struct_annotations
kgen.generator @read_struct_annotations() {
  // CHECK-NEXT: kgen.param.constant: param_list<type> = <[index, string]>
  kgen.param.constant: param_list<type> =
      <#kgen.struct_annotation_types<#annotated, -1> : !kgen.param_list<!kgen.type>>

  // The count a reader loops over is the list's size.
  // CHECK-NEXT: kgen.param.constant = <2>
  kgen.param.constant: index =
      <#kgen.param_list.size<:!kgen.param_list<!kgen.type>
        #kgen.struct_annotation_types<#annotated, -1> : !kgen.param_list<!kgen.type>>>

  // CHECK-NEXT: kgen.param.constant = <7>
  kgen.param.constant: !kgen.param<#kgen.param_list.get<:param_list<type>
      #kgen.struct_annotation_types<#annotated, -1> : !kgen.param_list<!kgen.type>, 0>>
      = <#kgen.struct_annotation<#annotated, -1, 0, !kgen.param_list<!kgen.type>>>

  // CHECK-NEXT: kgen.param.constant: string = <"tag">
  kgen.param.constant: !kgen.param<#kgen.param_list.get<:param_list<type>
      #kgen.struct_annotation_types<#annotated, -1> : !kgen.param_list<!kgen.type>, 1>>
      = <#kgen.struct_annotation<#annotated, -1, 1, !kgen.param_list<!kgen.type>>>
  kgen.return
}

// CHECK-LABEL: kgen.func @read_field_annotations
kgen.generator @read_field_annotations() {
  // CHECK-NEXT: kgen.param.constant: param_list<type> = <[index]>
  kgen.param.constant: param_list<type> =
      <#kgen.struct_annotation_types<#annotated, 0> : !kgen.param_list<!kgen.type>>

  // CHECK-NEXT: kgen.param.constant = <1>
  kgen.param.constant: !kgen.param<#kgen.param_list.get<:param_list<type>
      #kgen.struct_annotation_types<#annotated, 0> : !kgen.param_list<!kgen.type>, 0>>
      = <#kgen.struct_annotation<#annotated, 0, 0, !kgen.param_list<!kgen.type>>>

  // An unannotated field reads as an empty list rather than failing.
  // CHECK-NEXT: kgen.param.constant: param_list<type> = <[]>
  kgen.param.constant: param_list<type> =
      <#kgen.struct_annotation_types<#annotated, 1> : !kgen.param_list<!kgen.type>>
  // CHECK-NEXT: kgen.param.constant = <0>
  kgen.param.constant: index =
      <#kgen.param_list.size<:!kgen.param_list<!kgen.type>
        #kgen.struct_annotation_types<#annotated, 1> : !kgen.param_list<!kgen.type>>>
  kgen.return
}

// -----

// A struct with no annotations at all carries no annotation attributes, and
// still reads as an empty list.

kgen.struct.generator @Plain = struct_inst<"Plain"(value: index)>

#plain = #kgen.type<typevalue<:type #kgen.genref<@Plain>>,
                    struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.func @read_unannotated
kgen.generator @read_unannotated() {
  // CHECK-NEXT: kgen.param.constant = <0>
  kgen.param.constant: index =
      <#kgen.param_list.size<:!kgen.param_list<!kgen.type>
        #kgen.struct_annotation_types<#plain, -1> : !kgen.param_list<!kgen.type>>>
  kgen.return
}

// -----

// Annotations are written in the struct's own scope, so they can name its
// parameters and yield a different value per instantiation.

kgen.struct.generator @Param<N> = struct_inst<"Param"[N]<N>(value: index)>
    attributes {
  annotationTypes = #kgen<exprs[#kgen.type<index> : !kgen.type]>,
  annotations = #kgen<exprs[#kgen.param.decl.ref<"N"> : index]>
}

#param3 = #kgen.type<typevalue<:type #kgen.genref<@Param<3>>>,
                     struct<(index)>> : !kgen.type
#param5 = #kgen.type<typevalue<:type #kgen.genref<@Param<5>>>,
                     struct<(index)>> : !kgen.type

// CHECK-LABEL: kgen.func @read_per_instantiation
kgen.generator @read_per_instantiation() {
  // CHECK-NEXT: kgen.param.constant = <3>
  kgen.param.constant: !kgen.param<#kgen.param_list.get<:param_list<type>
      #kgen.struct_annotation_types<#param3, -1> : !kgen.param_list<!kgen.type>, 0>>
      = <#kgen.struct_annotation<#param3, -1, 0, !kgen.param_list<!kgen.type>>>
  // CHECK-NEXT: kgen.param.constant = <5>
  kgen.param.constant: !kgen.param<#kgen.param_list.get<:param_list<type>
      #kgen.struct_annotation_types<#param5, -1> : !kgen.param_list<!kgen.type>, 0>>
      = <#kgen.struct_annotation<#param5, -1, 0, !kgen.param_list<!kgen.type>>>
  kgen.return
}
