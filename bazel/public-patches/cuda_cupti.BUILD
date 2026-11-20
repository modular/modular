load("@rules_cc//cc:cc_library.bzl", "cc_library")

# CUPTI headers from the NVIDIA CUDA CUPTI redistributable package.
#
# cupti.h transitively includes <cuda.h> and <cuda_runtime_api.h> (via
# generated_cuda_runtime_api_meta.h), so this target depends on the
# cuda_cudart headers exposed by @//bazel/public-patches:cuda_runtime_headers.
cc_library(
    name = "cupti_headers",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = ["@//bazel/public-patches:cuda_runtime_headers"],
)

# The CUPTI runtime library itself. Nothing links against it -- libcupti_stub
# dlopen()s "libcupti.so" on first use -- but a container that wants GPU kernel
# tracing has to stage the real library somewhere the loader can find it. The
# archive ships `libcupti.so` -> `libcupti.so.<major>` -> `libcupti.so.<build>`;
# all three are exported so a consumer can dereference whichever it needs.
filegroup(
    name = "cupti_runtime",
    srcs = glob(["lib/libcupti.so*"]),
    visibility = ["//visibility:public"],
)
