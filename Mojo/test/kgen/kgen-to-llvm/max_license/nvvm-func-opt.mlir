// RUN: kgen --emit=llvm-opt -target-triple="nvptx64-nvidia-cuda" -target-cpu="sm_75" -target-features="" %s -o - | FileCheck %s

kgen.func export @thread_idx() -> !kgen.scalar<ui32> {
  // CHECK: %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %0 = pop.call_llvm_intrinsic "llvm.nvvm.read.ptx.sreg.tid.x", () : () -> !kgen.scalar<ui32>
  hlcf.return %0 : !kgen.scalar<ui32>
}
