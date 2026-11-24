; RUN: kgen-llvm-opt -O3 %s -S | FileCheck %s
;
; Test: Case where a callee uses an addrspace(3) global.
; This is a must pass for external/dynamic memory on Apple Silicon.
;
; Pattern:
;   @kernel directly uses @ext0, then calls @foo.
;   @foo uses @ext1, kernel never sees @ext1 in its own basic blocks.

target triple = "air64-apple-macosx"

@ext0 = external addrspace(3) global float, align 4
@ext1 = external addrspace(3) global float, align 4

; Both extern addrspace(3) globals must be gone from the output.
; CHECK-NOT: @ext0 = external{{.*}} addrspace(3)
; CHECK-NOT: @ext1 = external{{.*}} addrspace(3)

; Both must be replaced by addrspace(2) externally-initialized constants.
; CHECK-DAG: @ext0 = internal unnamed_addr addrspace(2) externally_initialized constant ptr addrspace(3) undef, align 8
; CHECK-DAG: @ext1 = internal unnamed_addr addrspace(2) externally_initialized constant ptr addrspace(3) undef, align 8

; @kernel uses @ext0 directly.
define void @kernel(ptr addrspace(1) %out) #0 {
  %v0 = load float, ptr addrspace(3) @ext0, align 4
  store float %v0, ptr addrspace(1) %out, align 4
  call void @foo(ptr addrspace(1) %out)
  ret void
}

; @foo uses @ext1, @kernel never touches @ext1.
; After promotion, @foo must load from the addrspace(2) constant, not old global.
define void @foo(ptr addrspace(1) %out) #1 {
  %v1 = load float, ptr addrspace(3) @ext1, align 4
  store float %v1, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: define void @foo(
; CHECK:       %ext1.tg = load ptr addrspace(3), ptr addrspace(2)
; CHECK-LABEL: define void @kernel(
; CHECK:       %ext0.tg = load ptr addrspace(3), ptr addrspace(2)

; Both globals must appear in !air.global_bindings, but both must use slot 0.
; First binding entry (ext0 or ext1, order may vary afaik).
; Each buffer node (slot marker i32 -1 at location_index 0) now prints before
; its air.global_binding node (llvm/llvm-project#216838 metadata print order).
; CHECK: !air.global_bindings
; CHECK: i32 -1
; CHECK: !"air.buffer"
; CHECK: !"air.location_index", i32 0
; CHECK: !"air.global_binding"
; Second binding must also be slot 0.
; CHECK: i32 -1
; CHECK: !"air.buffer"
; CHECK: !"air.location_index", i32 0
; CHECK: !"air.global_binding"
; No binding at slot 1.
; CHECK-NOT: !"air.location_index", i32 1

attributes #0 = { "metal.kernel"="true" "target-cpu"="apple-m3" }
attributes #1 = { noinline "target-cpu"="apple-m3" }
