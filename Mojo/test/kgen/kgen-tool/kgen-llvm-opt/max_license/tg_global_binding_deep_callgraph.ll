; RUN: kgen-llvm-opt -O3 %s -S | FileCheck %s
;
; Test: addrspace(3) globals referenced at arbitrary call depth. This is a must pass
; for enabling external/dynamic memory on Apple Silicon.
;
; Pattern:
;   @kernel1 uses @ext directly and calls @bar.
;   @bar calls @baz but never touches @ext itself.
;   @baz uses @ext directly.

target triple = "air64-apple-macosx"

@ext = external addrspace(3) global float, align 4

; Extern addrspace(3) global must be gone from the output.
; CHECK-NOT: @ext = external{{.*}} addrspace(3)

; Must be replaced by an addrspace(2) externally-initialized constant.
; CHECK: @ext = internal unnamed_addr addrspace(2) externally_initialized constant ptr addrspace(3) undef, align 8

; @kernel1 uses @ext directly.
define void @kernel1(ptr addrspace(1) %out) #0 {
  %v = load float, ptr addrspace(3) @ext, align 4
  store float %v, ptr addrspace(1) %out, align 4
  call void @bar(ptr addrspace(1) %out)
  ret void
}

; @bar is a pure pass-through; noinline keeps it visible in the output
define void @bar(ptr addrspace(1) %out) #1 {
  call void @baz(ptr addrspace(1) %out)
  ret void
}

; @baz uses @ext, @ext must be lifted out even though it is two calls away
define void @baz(ptr addrspace(1) %out) #1 {
  %v = load float, ptr addrspace(3) @ext, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

; @bar → @baz → @kernel1
; CHECK-LABEL: define void @bar(
; CHECK-NOT:   %ext.tg = load ptr addrspace(3)
; CHECK:       call void @baz(
; CHECK-LABEL: define void @baz(
; CHECK:       %ext.tg = load ptr addrspace(3), ptr addrspace(2)
; CHECK-LABEL: define void @kernel1(
; CHECK:       %ext.tg = load ptr addrspace(3), ptr addrspace(2)

; @ext must appear in !air.global_bindings bound to slot 0.
; The buffer node (slot marker i32 -1 at location_index 0) now prints before
; its air.global_binding node (llvm/llvm-project#216838 metadata print order).
; CHECK: !air.global_bindings
; CHECK: i32 -1
; CHECK: !"air.buffer"
; CHECK: !"air.location_index", i32 0
; CHECK: !"air.global_binding"

attributes #0 = { "metal.kernel"="true" "target-cpu"="apple-m3" }
attributes #1 = { noinline "target-cpu"="apple-m3" }
