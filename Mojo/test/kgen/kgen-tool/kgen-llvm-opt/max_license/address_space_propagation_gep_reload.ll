; RUN: kgen-llvm-opt -O3 %s -S | FileCheck %s --check-prefix=METAL
target triple = "air64-apple-macosx"

; MOCO-4913: a by-reference variadic pack reloads its second device pointer.
; METAL-LABEL: define void @array_gep_reload(
; METAL: store [2 x ptr addrspace(1)]
; METAL: load ptr addrspace(1), ptr
; METAL: load float, ptr addrspace(1)
define void @array_gep_reload(ptr %src, ptr %out) #0 {
  %slot = alloca [2 x ptr], align 8
  %a = insertvalue [2 x ptr] undef, ptr %src, 0
  %b = insertvalue [2 x ptr] %a, ptr %src, 1
  store [2 x ptr] %b, ptr %slot, align 8
  %elt = getelementptr [2 x ptr], ptr %slot, i64 0, i64 1
  %p = load ptr, ptr %elt, align 8
  %v = load float, ptr %p, align 4
  store float %v, ptr %out, align 4
  ret void
}

; Whole opaque payloads still promote through nested zero-offset wrappers.
; METAL-LABEL: define void @opaque_zero_gep_reload(
; METAL: load { ptr addrspace(1), { { i64 }, { {} } } }, ptr
; METAL: load float, ptr addrspace(1)
define void @opaque_zero_gep_reload({ i64, [1 x i64] } %blob, ptr %out) #0 {
  %slot = alloca { i64, [1 x i64] }, align 8
  store { i64, [1 x i64] } %blob, ptr %slot, align 8
  %outer = getelementptr { { { ptr, { { i64 }, { {} } } } } }, ptr %slot, i32 0, i32 0
  %inner = getelementptr { { ptr, { { i64 }, { {} } } } }, ptr %outer, i32 0, i32 0
  %payload = load { ptr, { { i64 }, { {} } } }, ptr %inner, align 8
  %p = extractvalue { ptr, { { i64 }, { {} } } } %payload, 0
  %v = load float, ptr %p, align 4
  store float %v, ptr %out, align 4
  ret void
}

; A partial reload takes each visible field's address space at its byte offset.
; METAL-LABEL: define void @visible_struct_fields(
; METAL: load { ptr addrspace(3), ptr addrspace(2) }, ptr
; METAL: load float, ptr addrspace(3)
; METAL: load float, ptr addrspace(2)
define void @visible_struct_fields({ i64, ptr addrspace(3), ptr addrspace(2) } %blob, ptr %out) #0 {
  %slot = alloca { i64, ptr addrspace(3), ptr addrspace(2) }, align 8
  store { i64, ptr addrspace(3), ptr addrspace(2) } %blob, ptr %slot, align 8
  %at = getelementptr i8, ptr %slot, i64 8
  %pair = load { ptr, ptr }, ptr %at, align 8
  %p = extractvalue { ptr, ptr } %pair, 0
  %q = extractvalue { ptr, ptr } %pair, 1
  %a = load float, ptr %p, align 4
  %b = load float, ptr %q, align 4
  %v = fadd float %a, %b
  store float %v, ptr %out, align 4
  ret void
}

; A reload crossing stored pointer boundaries provides no field match.
; METAL-LABEL: define void @unaligned_no_promote(
; METAL: load ptr, ptr
; METAL: load float, ptr %
define void @unaligned_no_promote([2 x ptr addrspace(1)] %blob, ptr %out) #0 {
  %slot = alloca [2 x ptr addrspace(1)], align 8
  store [2 x ptr addrspace(1)] %blob, ptr %slot, align 8
  %at = getelementptr i8, ptr %slot, i64 4
  %p = load ptr, ptr %at, align 4
  %v = load float, ptr %p, align 4
  store float %v, ptr %out, align 4
  ret void
}

; Dynamic offsets cannot identify a stored field.
; METAL-LABEL: define void @dynamic_no_promote(
; METAL: load ptr, ptr
; METAL: load float, ptr %
define void @dynamic_no_promote([2 x ptr addrspace(1)] %blob, ptr %out, i64 %idx) #0 {
  %slot = alloca [2 x ptr addrspace(1)], align 8
  store [2 x ptr addrspace(1)] %blob, ptr %slot, align 8
  %at = getelementptr [2 x ptr], ptr %slot, i64 0, i64 %idx
  %p = load ptr, ptr %at, align 8
  %v = load float, ptr %p, align 4
  store float %v, ptr %out, align 4
  ret void
}

; Array reloads require a common promoted element type.
; METAL-LABEL: define void @visible_array_reload(
; METAL: load [2 x ptr addrspace(3)], ptr
; METAL: load float, ptr addrspace(3)
define void @visible_array_reload({ i64, [2 x ptr addrspace(3)] } %blob, ptr %out) #0 {
  %slot = alloca { i64, [2 x ptr addrspace(3)] }, align 8
  store { i64, [2 x ptr addrspace(3)] } %blob, ptr %slot, align 8
  %at = getelementptr i8, ptr %slot, i64 8
  %a = load [2 x ptr], ptr %at, align 8
  %p = extractvalue [2 x ptr] %a, 1
  %v = load float, ptr %p, align 4
  store float %v, ptr %out, align 4
  ret void
}
; Whole-blob promotion also recovers opaque fields beside visible pointers.
; METAL-LABEL: define void @mixed_visible_opaque_reload(
; METAL: load { ptr addrspace(1), { { { ptr addrspace(1), { { i64 }, { {} } } } } } }, ptr
; METAL: load float, ptr addrspace(1)
; METAL: load float, ptr addrspace(1)
define void @mixed_visible_opaque_reload(
    { ptr addrspace(1), { i64, [1 x i64] } } %blob, ptr %out) #0 {
  %slot = alloca { ptr addrspace(1), { i64, [1 x i64] } }, align 8
  store { ptr addrspace(1), { i64, [1 x i64] } } %blob, ptr %slot, align 8
  %closure = load { ptr, { { { ptr, { { i64 }, { {} } } } } } }, ptr %slot, align 8
  %outer = extractvalue { ptr, { { { ptr, { { i64 }, { {} } } } } } } %closure, 0
  %inner = extractvalue { ptr, { { { ptr, { { i64 }, { {} } } } } } } %closure, 1, 0, 0, 0
  %a = load float, ptr %outer, align 4
  %b = load float, ptr %inner, align 4
  %v = fadd float %a, %b
  store float %v, ptr %out, align 4
  ret void
}
attributes #0 = { "metal.kernel"="true" }
