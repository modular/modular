; RUN: kgen-llvm-opt --disable-optimization-passes %s -o %t.air
; RUN: xcrun -sdk macosx air-objdump --disassemble %t.air | FileCheck %s

; REQUIRES: system-darwin, air-objdump

target triple = "air64-apple-macosx"

; CHECK: declare float @llvm.fma.f32(float, float, float) #[[ATTR0:[0-9]+]]

declare float @llvm.fma.f32(float, float, float)

; CHECK-LABEL: define void @call_with_contract()
define void @call_with_contract(){
  ; CHECK: call { float } @call_struct_return
  %1 = call contract { float } @call_struct_return()
  ret void
}

; CHECK-LABEL: define { float } @call_struct_return()
define { float } @call_struct_return() {
  ; CHECK: ret { float } zeroinitializer
  ret { float } zeroinitializer
}

; CHECK-LABEL: define void @test_debuginfo()
define void @test_debuginfo() {
    #dbg_value({ ptr, { [1 x i32], [1 x i32] }, { [1 x i32], [1 x i32] } } poison, !3, !DIExpression(), !11)
  ; CHECK: ret void, !dbg ![[DBG_LOC:.*]]
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_Mojo, file: !1, producer: "Mojo", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "bitcode_writer.ll", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocalVariable(name: "dummy", arg: 1, scope: !4, file: !1, line: 67, type: !9, flags: DIFlagPrivate)
!4 = distinct !DISubprogram(name: "bitcode_writer", linkageName: "test_debuginfo", scope: !1, file: !1, line: 66, type: !5, scopeLine: 66, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, annotations: !7)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{null, !9}
!7 = !{}
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "test_debuginfo", size: 192, align: 64, elements: !10)
!10 = !{}
!11 = !DILocation(line: 67, column: 5, scope: !4)

; CHECK: attributes #[[ATTR0]] = { nocallback nofree nosync nounwind readnone speculatable willreturn }

; CHECK: ![[DBG_LOC]] = !DILocation(line: 67, column: 5, scope: ![[DBG_SUBPROGRAM:.*]])
; CHECK: ![[DBG_SUBPROGRAM]] = distinct !DISubprogram(name: "bitcode_writer", linkageName: "test_debuginfo", scope: !{{.*}}, file: !{{.*}}, line: 66, type: ![[DBG_SUBROUTINE:.*]], scopeLine: 66, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !{{.*}}, annotations: !{{.*}})
; CHECK: ![[DBG_SUBROUTINE]] = !DISubroutineType(cc: DW_CC_normal, types: ![[DBG_SUBROUTINE_TYPE:.*]])
; CHECK: ![[DBG_SUBROUTINE_TYPE]] = !{null, ![[DBG_COMPOSITE:.*]]}
; CHECK: ![[DBG_COMPOSITE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "test_debuginfo", size: 192, align: 64, elements: !{{.*}})
