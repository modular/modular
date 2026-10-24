; Test that "+metal3_2,+air2_7_0" in target-features selects Metal 3.2 /
; AIR 2.7.0 / macOS 15.0 metadata.
;
; MetalAIRPass reads "target-features" from IR function attributes to derive
; the Metal/AIR version.
;
; RUN: kgen-llvm-opt -O0 %s -S | FileCheck %s

target triple = "air64-apple-macosx"

define dso_local void @my_kernel() #0 {
  ret void
}

attributes #0 = { norecurse "metal.kernel"="true" "target-cpu"="apple-m1" "target-features"="+metal3_2,+air2_7_0" }

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK: target triple = "air64-apple-macosx15.
; CHECK: !"Metal", i32 3, i32 2
; CHECK: i32 2, i32 7, i32 0
