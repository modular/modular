// RUN: kgen-opt %s -allow-unregistered-dialect | kgen-opt -allow-unregistered-dialect | FileCheck %s

// A builder whose components are all constant folds into the generator type it
// describes.
// CHECK-LABEL: "func_gen_type_builder.concrete"
// CHECK-SAME: () -> !kgen.generator<!lit.generator<[1](!lit.ref<index, imm *[0,0]>) throws -> index>>
"func_gen_type_builder.concrete"() : () -> !kgen.func_gen_type_builder<
  #kgen.param_list<> : !kgen.param_list<!kgen.type>,
  #kgen.param_list<[!lit.ref<index, imm #kgen.fn_gen_builder.param.decl.ref<"implicit_origin_0", !lit.origin<false>>>]> : !kgen.param_list<!kgen.type>,
  #kgen.type<index> : !kgen.type,
  #kgen.fn_metadata<[imm, imm], "throws", #lit.fn_meta_origin_data<1>>,
  // placeholder for the implicit origin declaration names. Should be remapped back to index based references.
  #kgen.param_list<"implicit_origin_0"> : !kgen.param_list<!kgen.string>>

// The argument and result types name the declared parameters, which the fold
// rewrites into index references of the generator it builds.
// CHECK-LABEL: "func_gen_type_builder.parametric"
// CHECK-SAME: () -> !kgen.generator<<type, type>(!kgen.param<*(0,0)>, !kgen.param<*(0,1)>) -> !kgen.param<*(0,0)>>
"func_gen_type_builder.parametric"() : () -> !kgen.func_gen_type_builder<
  #kgen.param_list<#kgen.fn_gen_builder.param.decl<"T", !kgen.type>, #kgen.fn_gen_builder.param.decl<"U", !kgen.type> : !kgen.type> : !kgen.param_list<!kgen.type>,
  #kgen.param_list<#kgen.fn_gen_builder.param.decl.ref<"T", !kgen.type>, #kgen.fn_gen_builder.param.decl.ref<"U", !kgen.type>> : !kgen.param_list<!kgen.type>,
  #kgen.fn_gen_builder.param.decl.ref<"T", !kgen.type>,
  #kgen.fn_metadata<[imm, imm], "none">,
  #kgen.param_list<> : !kgen.param_list<!kgen.string>>

// A parameter may be declared in terms of a parameter declared before it.
// CHECK-LABEL: "func_gen_type_builder.dependent_param"
// CHECK-SAME: () -> !kgen.generator<<type, *(0,0)>(!kgen.param<*(0,0)>) -> !kgen.param<*(0,0)>>
"func_gen_type_builder.dependent_param"() : () -> !kgen.func_gen_type_builder<
  #kgen.param_list<#kgen.fn_gen_builder.param.decl<"T", !kgen.type>, #kgen.fn_gen_builder.param.decl<"x", !kgen.param<#kgen.fn_gen_builder.param.decl.ref<"T", !kgen.type>>>> : !kgen.param_list<!kgen.type>,
  #kgen.param_list<#kgen.fn_gen_builder.param.decl.ref<"T", !kgen.type>> : !kgen.param_list<!kgen.type>,
  #kgen.fn_gen_builder.param.decl.ref<"T", !kgen.type>,
  #kgen.fn_metadata<[imm], "none">,
  #kgen.param_list<> : !kgen.param_list<!kgen.string>>

// Every component is a parameter expression, so a builder that is still
// symbolic survives until elaboration.
// CHECK-LABEL: "func_gen_type_builder.symbolic"
// CHECK-SAME: () -> !kgen.func_gen_type_builder
"func_gen_type_builder.symbolic"() : () -> !kgen.func_gen_type_builder<
  #kgen.param.decl.ref<"Ps"> : !kgen.param_list<!kgen.type>,
  #kgen.param.decl.ref<"Ts"> : !kgen.param_list<!kgen.type>,
  #kgen.param.decl.ref<"R"> : !kgen.type,
  #kgen.fn_metadata<[imm, mut], "throws">,
  #kgen.param_list<> : !kgen.param_list<!kgen.string>>
