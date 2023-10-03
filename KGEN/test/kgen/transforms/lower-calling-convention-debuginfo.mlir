// RUN: kgen-opt -lower-calling-convention %s -mlir-print-debuginfo | FileCheck %s

#file = #debuginfo.file<"foo.mlir" in "/">
#subprogram = #debuginfo.subprogram<
  compileUnit = #debuginfo.compile_unit<sourceLanguage = DW_LANG_C, file = #file, producer = "Mojo", isOptimized = true, emissionKind = Full>,
  scope = #file,
  name = "foo",
  linkageName = "foo",
  file = #file,
  line = 44,
  scopeLine = 44,
  subprogramFlags = "Definition|Optimized"
> : !debuginfo.subroutine<() -> (!kgen.none): DW_CC_normal>

// CHECK: !debuginfo.subroutine<() -> ()

#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.func @main() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none> loc(#loc)
  kgen.return %none : !kgen.none loc(#loc)
} loc(#loc)
