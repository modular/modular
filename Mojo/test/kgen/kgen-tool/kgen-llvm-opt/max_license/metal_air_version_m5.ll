; Test that "+metal4_0,+air2_8_0" in target-features selects Metal 4.0 /
; AIR 2.8.0 / macOS 26.0 metadata.
;
; MetalAIRPass reads "target-features" from IR function attributes to derive
; the Metal/AIR version.
; RUN: kgen-llvm-opt -O0 %s -S | FileCheck %s

target triple = "air64-apple-macosx"

define dso_local void @my_kernel() #0 {
  ret void
}

attributes #0 = { norecurse "metal.kernel"="true" "target-cpu"="apple-m5" "target-features"="+metal4_0,+air2_8_0" }

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: target triple = "air64-apple-macosx26.
; CHECK: !"Metal", i32 4, i32 0
; CHECK: i32 2, i32 8, i32 0
