// 1. Parse and print concrete, empty and parametric unions and member projections.
// 2. Round-trip types, operations, byte fragments and function symbols through bytecode.
// RUN: kgen-opt %s -verify-parameters -kgen-verifier | kgen-opt --kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt %s -emit-bytecode | kgen-opt -verify-parameters -kgen-verifier --kgen-print-inline-type-values | FileCheck %s

// CHECK-LABEL: @storage
// CHECK-SAME: untagged_union<i8, i64>
kgen.generator @storage(%arg: !kgen.struct<(untagged_union<i8, i64>)>) {
  hlcf.return
}

// CHECK-LABEL: @empty
// CHECK-SAME: untagged_union<>
kgen.generator @empty(%arg: !kgen.struct<(untagged_union<>)>) {
  hlcf.return
}

// CHECK-LABEL: @generic
// CHECK-SAME: untagged_union<[Ts]>
kgen.generator @generic<Ts: param_list<type>>(%arg: !kgen.struct<(untagged_union<[Ts]>)>) {
  hlcf.return
}

// CHECK-LABEL: @initialize
kgen.generator @initialize(%arg: !kgen.pointer<untagged_union<i8, i64>>) {
  // CHECK: kgen.untagged_union.init %arg0
  kgen.untagged_union.init %arg : !kgen.pointer<untagged_union<i8, i64>>
  hlcf.return
}

// CHECK-LABEL: @project
kgen.generator @project(%arg: !lit.ref<!kgen.untagged_union<i8, i64>, mut #lit.any.origin, 3>) {
  // CHECK: lit.ref.union.cast %arg0[1]
  %member = lit.ref.union.cast %arg[1] : !lit.ref<!kgen.untagged_union<i8, i64>, mut #lit.any.origin, 3> -> !lit.ref<i64, mut #lit.any.origin, 3>
  hlcf.return
}

// CHECK-LABEL: @generic_project
kgen.generator @generic_project<Ts: param_list<type>, I>(%arg: !lit.ref<!kgen.untagged_union<[Ts]>, imm #lit.any.origin>) {
  // CHECK: lit.ref.union.cast %arg0[I]
  %member = lit.ref.union.cast %arg[I] : !lit.ref<!kgen.untagged_union<[Ts]>, imm #lit.any.origin> -> !lit.ref<!kgen.param<#kgen.param_list.get<:param_list<type> Ts, I>>, imm #lit.any.origin>
  hlcf.return
}

// CHECK-LABEL: @byte_constant
kgen.generator @byte_constant() {
  // CHECK: kgen.param.constant: untagged_union<i8> = <#kgen.untagged_union<[5 : i8]>>
  %value = kgen.param.constant: !kgen.untagged_union<i8> = <#kgen.untagged_union<[5 : i8]> : !kgen.untagged_union<i8>>
  hlcf.return
}

kgen.func @callee() {
  hlcf.return
}

// CHECK-LABEL: @symbol_constant
kgen.generator @symbol_constant() {
  // Eight bytes place the function symbol after the i64 field.
  // CHECK: kgen.param.constant: untagged_union<struct<(i64, () -> ())>> = <#kgen.untagged_union<[5 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, #kgen.func.symbol<@callee> : !kgen.func<() -> ()>]>>
  %value = kgen.param.constant: !kgen.untagged_union<struct<(i64, () -> ())>> = <#kgen.untagged_union<[5 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, 0 : i8, #kgen.func.symbol<@callee> : !kgen.func<() -> ()>]> : !kgen.untagged_union<struct<(i64, () -> ())>>>
  hlcf.return
}
