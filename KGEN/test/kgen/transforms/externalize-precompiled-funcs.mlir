// RUN: kgen-opt -externalize-precompiled-functions -mlir-print-debuginfo %s | FileCheck %s

// CHECK: kgen.link dense_resource<somelib> : tensor<1xui8> as @aLib
kgen.link dense_resource<somelib> : tensor<1xui8> as @aLib

// CHECK: kgen.extern.func @precompiled() -> index from @aLib loc(#[[LOC:.+]])
kgen.func @precompiled() -> index attributes {precompiledBodyRef = @aLib} {
  %0 = kgen.param.constant = <5> loc(#loc0)
  kgen.return %0 : index loc(#loc0)
} loc(#loc0)

// CHECK-LABEL: kgen.func @main() -> index
kgen.func @main() -> index {
  // CHECK-NEXT: kgen.call @precompiled() : () -> index
  %0 = kgen.call @precompiled() : () -> index
  // CHECK-NEXT: kgen.return {{%[0-9]}} : index
  kgen.return %0 : index
}

// Make sure debug scope of extern function is correct.
!subroutine = !debuginfo.subroutine<() -> (index): DW_CC_normal>
#name = #debuginfo.source_name<(fn)"precompiled">
#file = #debuginfo.file<"test.mlir" in "/">
#cu = #debuginfo.compile_unit<sourceLanguage = DW_LANG_Mojo, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>
// CHECK: #[[SP:.+]] = #debuginfo.subprogram<
// CHECK-NOT: compileUnit =
// CHECK-NOT: Definition
#subprogram = #debuginfo.subprogram<compileUnit = #cu, scope = #file, name = #name, file = #file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized"> : !subroutine
// CHECK: #[[LOC]] = loc(fused<#[[SP]]>
#loc0 = loc(fused<#subprogram>["test.mlir":1:1])
