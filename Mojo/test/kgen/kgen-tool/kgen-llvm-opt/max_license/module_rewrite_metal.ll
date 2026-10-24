; RUN: kgen-llvm-opt -O3 %s -S | FileCheck %s
;
; Tests for the new InstructionRewrite transformations on the Metal backend:
;   - force-inline:                internal non-kernel functions gain alwaysinline
;   - stubUndefinedLibcCalls:      free() → erased; malloc() → null
;   - transformPrintBufferSentinels: __mojo_metal_os_log_64 → air.os_log
;   - static_string inlining:      static_string_* globals → alloca + store

target triple = "air64-apple-macosx"

; ---------------------------------------------------------------------------
; force-inline: internal non-kernel functions get alwaysinline
;
; The pass strips noinline and adds alwaysinline to all internal functions that
; are not Metal kernels, so air-lld can handle them without struct-pointer ABI
; issues.
;
; NOTE: MetalAIRPass outputs internal (non-kernel) functions before kernel
; functions, so this CHECK-LABEL block must precede the kernel checks below.
;
; CHECK: Function Attrs: alwaysinline
; CHECK-NEXT: define internal void @internal_helper(
; CHECK-NEXT:   ret void
define internal void @internal_helper(i32 %x) {
  ret void
}

; ---------------------------------------------------------------------------
; stubUndefinedLibcCalls: free() erased, malloc() → null pointer
;
; Metal GPU has no libc allocator. free() calls are removed outright;
; malloc() calls are replaced with a null pointer constant so callers
; that check the result handle it gracefully.
; ---------------------------------------------------------------------------

declare void @free(ptr)
declare ptr @malloc(i64)

; CHECK-LABEL: define void @test_stub_libc(
; CHECK-NOT:     call void @free(
; CHECK-NOT:     call ptr @malloc(
; CHECK:         store ptr null, ptr addrspace(1)
; CHECK:         ret void

; Declarations must be erased when no uses remain.
; CHECK-NOT: declare void @free(
; CHECK-NOT: declare ptr @malloc(
define void @test_stub_libc(ptr %p, ptr addrspace(1) %out) #0 {
  call void @free(ptr %p)
  %m = call ptr @malloc(i64 64)
  store ptr %m, ptr addrspace(1) %out, align 8
  ret void
}

; ---------------------------------------------------------------------------
; transformPrintBufferSentinels: __mojo_metal_os_log_64 → air.os_log
;
; _metal_print_write emits calls to a sentinel function with a 64-byte buffer.
; The pass replaces each call with:
;   1. An [64 x i32] alloca as the va-arg buffer.
;   2. 64 byte-load + zext + store sequences populating the va-arg buffer.
;   3. A call to air.os_log with the subsystem, category, format, and buffer.
; The sentinel declaration is erased when no uses remain.
; ---------------------------------------------------------------------------

declare void @__mojo_metal_os_log_64(ptr)

; CHECK-LABEL: define void @test_print_sentinel(
; CHECK:         alloca [64 x i32], align 4
; CHECK:         zext i8 {{.*}} to i32
; CHECK:         call void @air.os_log(
; CHECK-SAME:                         i32 1,
; CHECK-SAME:                         i64 256)
; CHECK-NOT:     call void @__mojo_metal_os_log_64(
; CHECK:         ret void

; CHECK-NOT: declare void @__mojo_metal_os_log_64(
define void @test_print_sentinel(ptr %buf) #0 {
  call void @__mojo_metal_os_log_64(ptr %buf)
  ret void
}

; ---------------------------------------------------------------------------
; force-inline: kernel calls to the internal helper survive in the output
; because the AlwaysInliner that runs after InstructionRewrite operates at
; module scope and may inline at link time. The observable pass effect is the
; alwaysinline attribute on the callee (already checked above).
; ---------------------------------------------------------------------------

; CHECK-LABEL: define void @test_force_inline(
; CHECK:         call void @internal_helper(
; CHECK:         ret void
define void @test_force_inline(i32 %x) #0 {
  call void @internal_helper(i32 %x)
  ret void
}

; ---------------------------------------------------------------------------
; static_string inlining: static_string_* globals → per-function alloca+store
;
; Metal's GPU JIT compiler can't always resolve global constant symbols, so
; the pass inlines static_string_* constant data into the function as a stack
; alloca, then stores the constant array into it before use.
; ---------------------------------------------------------------------------

@static_string_hello = private constant [6 x i8] c"hello\00", align 1

declare void @use_string(ptr)

; CHECK-LABEL: define void @test_static_string(
; CHECK:         alloca [6 x i8], align 16
; CHECK:         store [6 x i8] c"hello\00", ptr
; CHECK:         call void @use_string(ptr
; CHECK:         ret void
define void @test_static_string() #0 {
  call void @use_string(ptr @static_string_hello)
  ret void
}

; ---------------------------------------------------------------------------
; static_string inlining reached through a constant, not as a bare operand.
;
; A use is rarely the bare global. Mojo passes a string as a `{ptr, i64}` slice,
; so the global sits inside a constant struct, and an address-of plus an offset
; GEP folds into a constant expression. Every one of these forms has to be
; rewritten: a global left behind fails Metal pipeline creation at runtime with
; "Undefined symbols: static_string_<hash>".
; ---------------------------------------------------------------------------

@static_string_slice = private constant [6 x i8] c"slice\00", align 1
@static_string_offset = private constant [6 x i8] c"mojo!\00", align 1
@static_string_empty = private constant [1 x i8] zeroinitializer, align 1

declare void @use_slice({ ptr, i64 })

; A `{ptr, i64}` constant struct is rebuilt element by element so the pointer
; element can name the alloca.
; CHECK-LABEL: define void @test_static_string_in_struct(
; CHECK:         alloca [6 x i8], align 16
; CHECK:         store [6 x i8] c"slice\00", ptr
; CHECK:         insertvalue { ptr, i64 }
; CHECK-NOT:     @static_string_slice
; CHECK:         ret void
define void @test_static_string_in_struct() #0 {
  call void @use_slice({ ptr, i64 } { ptr @static_string_slice, i64 5 })
  ret void
}

; A constant-expression GEP at a nonzero offset becomes a GEP instruction on
; the alloca. (An offset-0 GEP folds to the bare global before we run.)
; CHECK-LABEL: define void @test_static_string_gep_offset(
; CHECK:         alloca [6 x i8], align 16
; CHECK:         store [6 x i8] c"mojo!\00", ptr
; CHECK:         getelementptr inbounds
; CHECK-NOT:     @static_string_offset
; CHECK:         ret void
define void @test_static_string_gep_offset() #0 {
  call void @use_string(ptr getelementptr inbounds ([6 x i8], ptr @static_string_offset, i64 0, i64 2))
  ret void
}

; An all-zero literal (the empty string) is a ConstantAggregateZero rather than
; a ConstantDataArray, and still has to be inlined.
; CHECK-LABEL: define void @test_static_string_empty(
; CHECK:         alloca [1 x i8], align 16
; CHECK:         store [1 x i8] zeroinitializer, ptr
; CHECK-NOT:     @static_string_empty
; CHECK:         ret void
define void @test_static_string_empty() #0 {
  call void @use_string(ptr @static_string_empty)
  ret void
}

attributes #0 = { "metal.kernel"="true" }
