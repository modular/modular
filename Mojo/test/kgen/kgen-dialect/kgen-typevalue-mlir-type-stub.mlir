// RUN: kgen-opt %s | FileCheck %s

// `TypeValueType::get` looks through the `__MLIRType[T]` stub to `T`, matching
// what `ParamType` folds to. A real struct whose MLIR type is `none` keeps its
// type value.

// CHECK-LABEL: @mlir_type_stub
// CHECK-SAME: %arg0: !kgen.pointer<none>
// CHECK-SAME: %arg1: !kgen.pointer<none>
// CHECK-SAME: %arg2: !kgen.pointer<typevalue<#kgen.genref<@"std::builtin::none::NoneType">>>
// CHECK-SAME: %arg3: !kgen.pointer<typevalue<#kgen.instref<@"std::builtin::none::NoneType">>>
kgen.func @mlir_type_stub(
  %arg0: !kgen.pointer<typevalue<[typevalue<#kgen.genref<@"std::builtin::_stubs::__MLIRType"<:non_struct_type none>>>, none]>>,
  %arg1: !kgen.pointer<typevalue<[typevalue<#kgen.instref<@"std::builtin::_stubs::__MLIRType,T=none">>, none]>>,
  %arg2: !kgen.pointer<typevalue<[typevalue<#kgen.genref<@"std::builtin::none::NoneType">>, none]>>,
  %arg3: !kgen.pointer<typevalue<[typevalue<#kgen.instref<@"std::builtin::none::NoneType">>, none]>>
) {
  hlcf.return
}
