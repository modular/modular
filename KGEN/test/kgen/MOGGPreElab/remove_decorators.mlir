// RUN: kgen-opt %s --remove-mogg-decorators | FileCheck %s

kgen.generator @"register::register::mogg_tensor_allocator()"() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator @test_alloc()  -> !kgen.none
  decorators <:none apply(:() -> !kgen.none @"register::register::mogg_tensor_allocator()")> {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator @test_alloc() -> !kgen.none attributes {"register::register::mogg_tensor_allocator"}


kgen.generator @"register::register::mogg_elementwise()"() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator export @elementwise() -> !kgen.none
  decorators <
    :() -> !kgen.none @"register::register::mogg_elementwise()"> {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator export @elementwise() -> !kgen.none attributes {"register::register::mogg_elementwise"} {


kgen.generator @"register::register::mogg_register(stdlib::builtin::string_literal::StringLiteral)"(%arg0: !kgen.string borrow) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator export @kernel() -> !kgen.none
  decorators <
    :none apply(:(!kgen.string borrow) -> !kgen.none @"register::register::mogg_register(stdlib::builtin::string_literal::StringLiteral)", "mo.pow")> {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL: kgen.generator export @kernel() -> !kgen.none attributes {_mogg_kernel = ["mo.pow" : !kgen.string, -1]} {

kgen.generator @"register::register::mogg_register_override(stdlib::builtin::string_literal::StringLiteral,stdlib::builtin::int::Int)"(%arg0: !kgen.string borrow, %arg1: index borrow) -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

kgen.generator export @kernel_override() -> !kgen.none
  decorators <
     :none apply(:(!kgen.string borrow, index borrow) -> !kgen.none @"register::register::mogg_register_override(stdlib::builtin::string_literal::StringLiteral,stdlib::builtin::int::Int)", "kernel_override", 4242)> {
  %none = kgen.param.constant: none = <#kgen.none>
  kgen.return %none : !kgen.none
}

// CHECK-LABEL:   kgen.generator export @kernel_override() -> !kgen.none attributes {_mogg_kernel = ["kernel_override" : !kgen.string, 4242 : index]} {
