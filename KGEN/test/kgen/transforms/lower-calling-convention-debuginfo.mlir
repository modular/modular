// RUN: kgen-opt -lower-calling-conventions %s -mlir-print-debuginfo | FileCheck %s

#subprogram = #debuginfo.subprogram<name = <"foo">> : !debuginfo.subroutine<() -> (!kgen.none): DW_CC_normal>

// CHECK: !debuginfo.subroutine<() -> ()

#loc = loc(fused<#subprogram>["foo.mlir":0:0])

kgen.func @main() -> !kgen.none {
  %none = kgen.param.constant: none = <#kgen.none> loc(#loc)
  kgen.return %none : !kgen.none loc(#loc)
} loc(#loc)
