; Small test module containing llvm.compiler.used referencing two globals.
source_filename = "bitcode_llvm_compiler_used.ll"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.nvshmemi_device_host_state_v1 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, ptr, i64, ptr, ptr, i8, i8, i32, ptr, ptr, ptr, ptr, ptr, %struct.gpu_coll_env_params_v2, ptr, ptr, ptr, ptr, i64, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i32, ptr, ptr, i8, i8, i8 }
%struct.gpu_coll_env_params_v2 = type { i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i32, i64, i32, i32, i64, i64, [416 x i8] }
%struct.nvshmemi_version_t = type { i32, i32, i32 }

; This is a special global variable that should be included in every submodule.
; This example is taken from the nvshmem library. It tracks a few version 
; numbers that might be needed by their backend or their runtime.
; According to the LLVM lang ref, the compiler must assume that there's an
; invisible reference to these global variables that is keeping them alive, so
; the semantics-preserving solution is to have it included in each submodule
; since each submodule is supposed to be able to be used individually.

@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(4) @nvshmemi_device_lib_version_d to ptr), ptr addrspacecast (ptr addrspace(4) @nvshmemi_device_state_d to ptr)], section "llvm.metadata"
@nvshmemi_device_state_d = dso_local addrspace(4) externally_initialized global %struct.nvshmemi_device_host_state_v1 zeroinitializer, align 8
@nvshmemi_device_lib_version_d = dso_local addrspace(4) externally_initialized global %struct.nvshmemi_version_t { i32 3, i32 3, i32 20 }, align 4

define dso_local ptx_kernel void @my_i32_inc(ptr noundef %0) {
  %2 = load i32, ptr %0, align 4
  %3 = add i32 %2, 1
  store i32 %3, ptr %0, align 4
  ret void
}

define dso_local ptx_kernel void @my_i32_dec(ptr noundef %0) {
  %2 = load i32, ptr %0, align 4
  %3 = sub i32 %2, 1
  store i32 %3, ptr %0, align 4
  ret void
}
