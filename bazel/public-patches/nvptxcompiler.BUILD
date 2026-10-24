load("@//bazel:api.bzl", "modular_shared_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")

# Replace ctors with init_array.
#
# Workaround for:
# - https://forums.developer.nvidia.com/t/ctors-sections-in-static-libraries/285542
#
# References:
# - https://github.com/NVIDIA/TensorRT-Incubator/blob/cd419f7c74cdb8b82e21f46f9e1095b5f19e6204/mlir-tensorrt/executor/cmake/ExecutorDependencies.cmake#L185
# - https://github.com/llvm/llvm-project/issues/68071#issuecomment-1848695832
# - https://github.com/llvm/llvm-project/issues/30572
# - https://maskray.me/blog/2021-11-07-init-ctors-init-array
genrule(
    name = "nvptxcompiler_rewritten",
    srcs = ["lib/libnvptxcompiler_static.a"],
    outs = ["rewritten_libnvptxcompiler_static.a"],
    cmd = "$(location @//bazel/internal/cc-toolchain/tools:llvm-objcopy) --rename-section .ctors=.init_array --rename-section .dtors=.fini_array $(location lib/libnvptxcompiler_static.a) $(location rewritten_libnvptxcompiler_static.a)",
    tools = ["@//bazel/internal/cc-toolchain/tools:llvm-objcopy"],
)

cc_import(
    name = "nvptxcompiler",
    static_library = ":nvptxcompiler_rewritten",
    alwayslink = True,
)

modular_shared_library(
    name = "NVPTX",
    srcs = [],
    hdrs = ["include/nvPTXCompiler.h"],
    includes = ["include"],
    visibility = [
        "@//bazel:__subpackages__",
        "@//oss/modular/bazel:__subpackages__",
        "@//utils:__subpackages__",
    ],
    # CUDAHeaders provides cuda.h from the NVIDIA cuda_cudart redistributable
    # so that targets depending on NVPTX automatically get the CUDA driver
    # API header.
    deps = [
        ":nvptxcompiler",
        "@//bazel/public-patches:CUDAHeaders",
    ],
)
