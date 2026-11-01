; Reduced from bench_rms_norm_fused_fp8.mojo using llvm-reduce.
;
; When address space propagation updates the callee's pointer parameter to
; addrspace(1), the call site may still pass a value with the old type (e.g.
; a thread-local alloca). Verify that updateCall inserts an addrspacecast to
; reconcile the type mismatch.
;
; RUN: kgen-llvm-opt -O3 %s -S | FileCheck %s
target triple = "air64-apple-macosx"

define void @callee(ptr %0) {
  ret void
}

; CHECK-LABEL: define void @caller(
; CHECK:    [[CAST:%.*]] = bitcast ptr %{{.*}} to ptr
; CHECK:    call void @callee(ptr [[CAST]])
define void @caller(ptr %0) #0 {
  %alloca = alloca float, i64 16, align 4
  %2 = call i32 @air.threadgroup_position_in_grid.x()
  call void @callee(ptr %alloca)
  ret void
}

declare i32 @air.threadgroup_position_in_grid.x()

attributes #0 = { "metal.kernel"="true" }
