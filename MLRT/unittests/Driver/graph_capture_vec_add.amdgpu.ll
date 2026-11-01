target triple = "amdgcn-amd-amdhsa"

; Read workgroup_size_x from the AQL dispatch packet (offset 4, uint16_t).
; @llvm.amdgcn.workgroup.size.x is not lowered by the AMDGPU backend; use
; dispatch.ptr instead per the AMD HSA spec.
define amdgpu_kernel void @vec_add(ptr addrspace(1) %out,
                                    ptr addrspace(1) %in0,
                                    ptr addrspace(1) %in1,
                                    i64 %len) {
entry:
  %tid.x     = call i32 @llvm.amdgcn.workitem.id.x()
  %bid.x     = call i32 @llvm.amdgcn.workgroup.id.x()
  %disp      = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %bdim.ptr  = getelementptr i8, ptr addrspace(4) %disp, i32 4
  %bdim.i16  = load i16, ptr addrspace(4) %bdim.ptr, align 2
  %bdim.x    = zext i16 %bdim.i16 to i32
  %base      = mul i32 %bid.x, %bdim.x
  %idx.i32   = add i32 %base, %tid.x
  %idx       = zext i32 %idx.i32 to i64
  %in_range  = icmp ult i64 %idx, %len
  br i1 %in_range, label %body, label %exit

body:
  %p.in0 = getelementptr float, ptr addrspace(1) %in0, i64 %idx
  %p.in1 = getelementptr float, ptr addrspace(1) %in1, i64 %idx
  %p.out = getelementptr float, ptr addrspace(1) %out, i64 %idx
  %v0    = load float, ptr addrspace(1) %p.in0, align 4
  %v1    = load float, ptr addrspace(1) %p.in1, align 4
  %sum   = fadd float %v0, %v1
  store float %sum, ptr addrspace(1) %p.out, align 4
  br label %exit

exit:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
