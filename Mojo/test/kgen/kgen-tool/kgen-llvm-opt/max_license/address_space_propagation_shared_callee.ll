; RUN: kgen-llvm-opt -O3 %s -S | FileCheck %s --check-prefix=METAL
target triple = "air64-apple-macosx"

; Regression test for MetalAIRPass::promoteCalleeBlobLoads: a device-blob-
; reloading callee SHARED across call sites (multiple uses) must still have its
; reload promoted to addrspace(1). This is the fused-concat pattern -- the
; outlined per-element lambda reloading the `_FusedInputVariadicTensors` blob
; is shared across concat sites. A `hasOneUse()` guard once skipped these,
; silently zeroing `mo.concat` with >= 4 operands on Metal.
;
; The callee below is called twice (fails `hasOneUse()`) yet must still be
; promoted to addrspace(1).

define internal void @shared_byptr_callee(ptr noundef nonnull %slot,
                                          ptr noundef %out) noinline {
  ; Reload the spilled struct; the ptr field must become addrspace(1).
  %s = load { ptr, ptr }, ptr %slot, align 8
  %p0 = extractvalue { ptr, ptr } %s, 0
  %v = load float, ptr %p0, align 4
  store float %v, ptr %out, align 4
  ret void
}

; METAL-LABEL: define void @kernel_two_calls(
; METAL:         store { ptr addrspace(1), ptr addrspace(1) }
; METAL:         call void @shared_byptr_callee(
; METAL:         store { ptr addrspace(1), ptr addrspace(1) }
; METAL:         call void @shared_byptr_callee(
define void @kernel_two_calls(ptr noundef %in0, ptr noundef %in1,
                              ptr noundef %out0, ptr noundef %out1) #0 {
  %slotA = alloca { ptr, ptr }, align 8
  %a0 = insertvalue { ptr, ptr } undef, ptr %in0, 0
  %a1 = insertvalue { ptr, ptr } %a0, ptr %in1, 1
  store { ptr, ptr } %a1, ptr %slotA, align 8
  call void @shared_byptr_callee(ptr %slotA, ptr %out0)
  %slotB = alloca { ptr, ptr }, align 8
  %b0 = insertvalue { ptr, ptr } undef, ptr %in0, 0
  %b1 = insertvalue { ptr, ptr } %b0, ptr %in1, 1
  store { ptr, ptr } %b1, ptr %slotB, align 8
  call void @shared_byptr_callee(ptr %slotB, ptr %out1)
  ret void
}

attributes #0 = { "metal.kernel"="true" }

; The shared callee's reload must still come back as device pointers.
; METAL-LABEL: define internal void @shared_byptr_callee(
; METAL:         load { ptr addrspace(1), ptr addrspace(1) },
; METAL:         load float, ptr addrspace(1)
