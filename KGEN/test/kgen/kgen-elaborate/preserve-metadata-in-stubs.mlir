// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" | FileCheck %s

// Test that @__llvm_metadata is preserved when a kernel is passed through
// a constrained function type parameter (MOCO-3054).
//
// When using enqueue_function_experimental[kernel](), the kernel is passed
// through a constrained function type like `fn() -> None`. This creates a
// wrapper/thunk that calls the original kernel. The metadata from the original
// kernel must be propagated to the wrapper for correct GPU execution.

// Original kernel with nvvm.maxntid metadata
kgen.generator @kernel_with_metadata() attributes {
  LLVMMetadataArray = [
    "nvvm.maxntid", #pop.array<128> : !pop.array<1, i32>
  ]
} {
  kgen.return
}

// Wrapper function that calls the original kernel (simulates constrained fn type)
// The wrapper has no metadata itself - metadata must be propagated from callee
kgen.generator @wrapper_calling_kernel() {
  kgen.call @kernel_with_metadata() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func export @test_metadata_propagation
// The LLVM IR should contain nvvm.maxntid in the function attributes
// CHECK: nvvm.maxntid
kgen.generator export @test_metadata_propagation() {
  kgen.param.declare nvptx: target = <#kgen.target<
    triple = "nvptx64-nvidia-cuda",
    arch = "sm_75",
    data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",
    simd_bit_width = 128
  >>

  // Compile the wrapper function via compile_offload.
  // The extractCalleeMetadata helper in KGENCompiler.cpp walks the wrapper's
  // body to find the callee (kernel_with_metadata) and extracts its metadata,
  // which is then applied to the wrapper entry point.
  %0 = kgen.compile_offload<nvptx, 2, "", :() -> () @wrapper_calling_kernel>
                            : !kgen.struct<(string, string, index, pointer<none>)>
  kgen.return
}

// =============================================================================
// Test: Fallback to wrapper's own metadata when callee has none
// =============================================================================

// Kernel without any metadata
kgen.generator @kernel_without_metadata() {
  kgen.return
}

// Wrapper with its own metadata calling a kernel without metadata
kgen.generator @wrapper_with_own_metadata() attributes {
  LLVMMetadataArray = [
    "nvvm.maxntid", #pop.array<64> : !pop.array<1, i32>
  ]
} {
  kgen.call @kernel_without_metadata() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func export @test_fallback_to_wrapper_metadata
// When callee has no metadata, wrapper's own metadata should be preserved
// CHECK: nvvm.maxntid
kgen.generator export @test_fallback_to_wrapper_metadata() {
  kgen.param.declare nvptx: target = <#kgen.target<
    triple = "nvptx64-nvidia-cuda",
    arch = "sm_75",
    data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",
    simd_bit_width = 128
  >>

  %0 = kgen.compile_offload<nvptx, 2, "", :() -> () @wrapper_with_own_metadata>
                            : !kgen.struct<(string, string, index, pointer<none>)>
  kgen.return
}

// =============================================================================
// Test: Callee metadata takes precedence over wrapper metadata
// =============================================================================

// Wrapper has maxntid=32 but calls kernel with maxntid=128
// Callee's metadata should take precedence
kgen.generator @wrapper_with_conflicting_metadata() attributes {
  LLVMMetadataArray = [
    "nvvm.maxntid", #pop.array<32> : !pop.array<1, i32>
  ]
} {
  kgen.call @kernel_with_metadata() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func export @test_callee_metadata_precedence
// Callee's maxntid=128 should override wrapper's maxntid=32
// CHECK: nvvm.maxntid
// CHECK-SAME: 128
kgen.generator export @test_callee_metadata_precedence() {
  kgen.param.declare nvptx: target = <#kgen.target<
    triple = "nvptx64-nvidia-cuda",
    arch = "sm_75",
    data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",
    simd_bit_width = 128
  >>

  %0 = kgen.compile_offload<nvptx, 2, "", :() -> () @wrapper_with_conflicting_metadata>
                            : !kgen.struct<(string, string, index, pointer<none>)>
  kgen.return
}

// =============================================================================
// Test: Nested wrappers - metadata propagates through chain
// =============================================================================

// Inner wrapper calls kernel with metadata
kgen.generator @inner_wrapper() {
  kgen.call @kernel_with_metadata() : () -> ()
  kgen.return
}

// Outer wrapper calls inner wrapper (no direct call to kernel)
kgen.generator @outer_wrapper() {
  kgen.call @inner_wrapper() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.func export @test_nested_wrapper
// Note: With current implementation, only immediate callee is checked.
// The outer wrapper calls inner_wrapper which has no metadata, so we
// fall back to outer_wrapper's metadata (which is empty).
// This test documents the current behavior - metadata does NOT propagate
// through nested wrappers.
// CHECK-NOT: nvvm.maxntid
kgen.generator export @test_nested_wrapper() {
  kgen.param.declare nvptx: target = <#kgen.target<
    triple = "nvptx64-nvidia-cuda",
    arch = "sm_75",
    data_layout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",
    simd_bit_width = 128
  >>

  %0 = kgen.compile_offload<nvptx, 2, "", :() -> () @outer_wrapper>
                            : !kgen.struct<(string, string, index, pointer<none>)>
  kgen.return
}

// Note: Arg metadata propagation is tested in the Mojo integration test
// (llvm_metadata_nvptx.mojo) with nvvm.grid_constant, since MLIR tests go
// through full LLVM lowering which requires valid LLVM attributes.
