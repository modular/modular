; RUN: kgen-llvm-opt -O3 %s -S | FileCheck %s --check-prefix=METAL
target triple = "air64-apple-macosx"

; Tests for the kgen.variant<T, i1> alloca type-pun pattern in MetalAIRPass.
;
; kgen.variant<T, i1> lowers to { { i64, [N x i8] }, i8 } -- an opaque byte
; array that hides T's pointer fields from createTypeWithAddressSpace. A
; callee unpacks T by storing the inner payload to an alloca and reloading as
; the transparent T struct; the ptr fields emerge as addrspace(0). The
; type-pun scan in MetalAIRPass promotes these loads after the main worklist
; propagation pass has populated functionsToUpdate.
;
; Model types:
;   opaque   : { i64, [8 x i8] }   -- 16 bytes, no ptr fields
;   transparent T : { ptr, i64 }    -- 16 bytes, ptr field at offset 0
;
; NOTE: MetalAIRPass emits kernel functions before internal callees, so
; METAL-LABEL checks for @type_pun_callee appear at the bottom of this file.

; ---------------------------------------------------------------------------
; Test 1: kernel-inline type-pun.
; The kernel entry is in functionsToUpdate from seeding. The alloca storing
; the opaque blob and reloading as the transparent T struct lives directly in
; the kernel body.
; ---------------------------------------------------------------------------

; METAL-LABEL: define void @kernel_inline_type_pun(
; METAL:         load { ptr addrspace(1), i64 },
; METAL:         store i32 42, ptr addrspace(1)
define void @kernel_inline_type_pun({ i64, [8 x i8] } noundef %closure_body) #0 {
  %slot = alloca { i64, [8 x i8] }, align 8
  store { i64, [8 x i8] } %closure_body, ptr %slot
  ; Type-pun reload: ptr field must be promoted to addrspace(1).
  %inner = load { ptr, i64 }, ptr %slot
  %p = extractvalue { ptr, i64 } %inner, 0
  store i32 42, ptr %p, align 4
  ret void
}

; ---------------------------------------------------------------------------
; Test 2: inter-procedural type-pun.
; A kernel passes both a direct device ptr and an opaque blob to a callee.
; The direct device ptr causes the callee to be added to functionsToUpdate
; during the main worklist pass. The type-pun scan then detects the alloca
; in the callee and promotes the hidden ptr to addrspace(1).
;
; Note: @type_pun_callee is an internal function; its METAL-LABEL check
; appears at the end of this file (after the kernel LABEL checks) to match
; the output order where MetalAIRPass emits kernels before internal callees.
; ---------------------------------------------------------------------------

define internal void @type_pun_callee(
    ptr %dev_ptr,
    { i64, [8 x i8] } %blob) noinline {
  ; dev_ptr is promoted to addrspace(1) by the main worklist pass (the kernel
  ; passes a device ptr here), which inserts this callee into functionsToUpdate.
  ; blob has no visible ptr fields so normal propagation cannot promote it.
  %slot = alloca { i64, [8 x i8] }, align 8
  store { i64, [8 x i8] } %blob, ptr %slot
  ; Type-pun: same alloca reloaded as transparent struct.
  ; The ptr field must be promoted to addrspace(1) by the type-pun scan.
  %inner = load { ptr, i64 }, ptr %slot
  %p = extractvalue { ptr, i64 } %inner, 0
  store i32 42, ptr %p, align 4
  store i32 99, ptr %dev_ptr, align 4
  ret void
}

; METAL-LABEL: define void @type_pun_kernel_interprocedural(
; METAL:         call void @type_pun_callee(ptr addrspace(1)
define void @type_pun_kernel_interprocedural(
    ptr %dev_ptr,
    { i64, [8 x i8] } %blob) #0 {
  call void @type_pun_callee(ptr %dev_ptr, { i64, [8 x i8] } %blob)
  ret void
}

; ---------------------------------------------------------------------------
; Test 3: partial load must NOT be promoted (size mismatch).
; The opaque store is 16 bytes but the load is only 8 bytes ({ ptr }). The
; size check in the type-pun scan must suppress promotion.
; ---------------------------------------------------------------------------

; METAL-LABEL: define void @partial_load_no_promote(
; METAL:         store { i64, [8 x i8] }
; METAL-NOT:     load { ptr addrspace(1) }
; METAL:         ret void
define void @partial_load_no_promote(ptr %dev_ptr) #0 {
  %slot = alloca { i64, [8 x i8] }, align 8
  ; Opaque blob store: 16 bytes.
  store { i64, [8 x i8] } zeroinitializer, ptr %slot
  ; Partial load: 8 bytes -- smaller than the 16-byte opaque store.
  ; The size mismatch must prevent promotion so %p stays as ptr (addrspace 0).
  %partial = load { ptr }, ptr %slot
  %p = extractvalue { ptr } %partial, 0
  ; Make the result observable so it is not optimized away.
  store i32 42, ptr %p, align 4
  ret void
}

; ---------------------------------------------------------------------------
; Test 4: inter-procedural alloca-by-pointer type-pun.
; A kernel inserts device pointers into a transparent struct, spills it to an
; alloca, and passes the alloca BY POINTER to an outlined callee. The callee
; reloads the struct from its (addrspace(0)) parameter and reads the pointer
; fields -- which, without promotion, emerge as addrspace(0) and read zeros on
; Metal. This is the fused-variadic concat pattern: the per-element lambda
; receives the `_FusedInputVariadicTensors` (holding addrspace(1) input ptrs)
; by pointer. The interprocedural type-pun scan must promote the callee's
; reload to addrspace(1). The store value here already carries addrspace(1)
; pointers (promoted from the kernel args), so this exercises the
; promoted-store-type branch of the scan.
; ---------------------------------------------------------------------------

define internal void @alloca_byptr_callee(ptr noundef nonnull %slot,
                                          ptr noundef %out) noinline {
  ; Reload the spilled struct; the ptr field must become addrspace(1).
  %s = load { ptr, ptr }, ptr %slot, align 8
  %p0 = extractvalue { ptr, ptr } %s, 0
  %v = load float, ptr %p0, align 4
  store float %v, ptr %out, align 4
  ret void
}

; METAL-LABEL: define void @alloca_byptr_kernel(
; METAL:         store { ptr addrspace(1), ptr addrspace(1) }
; METAL:         call void @alloca_byptr_callee(
define void @alloca_byptr_kernel(ptr noundef %in0, ptr noundef %in1,
                                 ptr noundef %out) #0 {
  %slot = alloca { ptr, ptr }, align 8
  %s0 = insertvalue { ptr, ptr } undef, ptr %in0, 0
  %s1 = insertvalue { ptr, ptr } %s0, ptr %in1, 1
  store { ptr, ptr } %s1, ptr %slot, align 8
  call void @alloca_byptr_callee(ptr %slot, ptr %out)
  ret void
}

attributes #0 = { "metal.kernel"="true" }

; ---------------------------------------------------------------------------
; @type_pun_callee checks: MetalAIRPass emits internal functions after all
; kernel entry functions, so these checks must follow the kernel LABEL checks.
; ---------------------------------------------------------------------------

; The interprocedural alloca-by-pointer callee reloads its parameter as a
; struct of device pointers and dereferences one in the device address space.
; MetalAIRPass emits this callee before @type_pun_callee.
; METAL-LABEL: define internal void @alloca_byptr_callee(
; METAL:         load { ptr addrspace(1), ptr addrspace(1) },
; METAL:         load float, ptr addrspace(1)

; METAL-LABEL: define internal void @type_pun_callee(
; METAL:         load { ptr addrspace(1), i64 },
; METAL:         extractvalue { ptr addrspace(1), i64 }
; METAL:         store i32 42, ptr addrspace(1)
