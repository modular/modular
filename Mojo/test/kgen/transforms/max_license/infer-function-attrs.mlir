// RUN: kgen-opt %s -infer-function-attrs -allow-unregistered-dialect | FileCheck %s

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
  // CHECK: kgen.func @convergent_call() attributes {convergent}
  kgen.func @convergent_call() {
    nvvm.barrier
    hlcf.return
  }

  // CHECK: @test_convergent_call() attributes {convergent}
  kgen.func @test_convergent_call() {
    kgen.call @convergent_call() : () -> ()
    hlcf.return
  }
}
