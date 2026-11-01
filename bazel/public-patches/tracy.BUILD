load("@//bazel:api.bzl", "modular_shared_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

_COPTS = [
    "-fexceptions",
    "-fvisibility=default",
    "-Wno-missing-field-initializers",
    "-Wno-ctad-maybe-unsupported",
    "-Wno-ignored-qualifiers",
    "-Wno-reorder-ctor",
]

# Upstream 0.14 capture.cpp prints size_t with PRIu64; fails -Wformat on
# macOS where size_t is unsigned long. Scoped to the example binaries so the
# client/server library builds stay format-checked.
_TOOL_COPTS = _COPTS + ["-Wno-format"]

cc_library(
    name = "tracy_no_on_demand",
    testonly = True,
    srcs = ["public/TracyClient.cpp"],
    hdrs = [
        "public/tracy/Tracy.hpp",
        "public/tracy/TracyC.h",
        "public/tracy/TracyCUDA.hpp",
    ],
    copts = _COPTS,
    defines = [
        # NOTE: No TRACY_ON_DEMAND here
        "TRACY_ENABLE",
    ] + select({
        "@//:tracy_rocm_enabled": ["TRACY_ROCPROF"],
        "//conditions:default": [],
    }),
    includes = [
        "public",
        "server",
    ],
    textual_hdrs = glob([
        "public/client/**/*.cpp",
        "public/client/**/*.h",
        "public/client/**/*.hpp",
        "public/common/**/*.cpp",
        "public/common/**/*.h",
        "public/common/**/*.hpp",
        "public/libbacktrace/**/*.cpp",
        "public/libbacktrace/**/*.h",
        "public/libbacktrace/**/*.hpp",
        "server/**/*.hpp",
        "server/**/*.h",
    ]),
    visibility = ["@//Support/unittests:__subpackages__"],
    deps = ["@ppqsort"] + select({
        # Only include CUPTI and the CUDA stubs when Tracy profiling is
        # explicitly enabled.  This avoids requiring libcupti.so / libcuda.so.1
        # / libcudart.so for regular builds.
        "@//:tracy_cuda_enabled": [
            "@//Support:libcuda_stub",
            "@//Support:libcudart_stub",
            "@//Support:libcupti_stub",
        ],
        "//conditions:default": [],
    }) + select({
        # Only include the rocprofiler-sdk stub when Tracy profiling is
        # explicitly enabled with an AMD GPU.  This avoids requiring
        # librocprofiler-sdk.so for regular builds.
        "@//:tracy_rocm_enabled": [
            "@//Support:librocprofiler_tracy_stub",
            "@//bazel/public-patches:rocprofiler_headers",
        ],
        "//conditions:default": [],
    }),
)

# Build Tracy client into its own shared library to ensure there is exactly one
# copy of Tracy's runtime (rpmalloc, profiler) per process. All users that need
# Tracy at runtime should link against this DSO instead of the static client.
modular_shared_library(
    name = "Tracy",
    srcs = [
        "public/TracyClient.cpp",
    ] + glob(
        [
            "getopt/**/*.c",
            "getopt/**/*.h",
            "public/client/**/*.h",
            "public/client/**/*.hpp",
            "public/common/**/*.cpp",
            "public/common/**/*.h",
            "public/common/**/*.hpp",
            "public/libbacktrace/**/*.h",
            "public/libbacktrace/**/*.hpp",
            "server/**/*.cpp",
            "server/**/*.h",
            "server/**/*.hpp",
        ],
        exclude = [
            # These are the files that are #included in TracyClient.cpp,
            # including them in srcs results in duplicate symbols
            "public/client/tracy_rpmalloc.cpp",
            "public/client/TracyAlloc.cpp",
            "public/client/TracyCallstack.cpp",
            "public/client/TracyDxt1.cpp",
            "public/client/TracyKCore.cpp",
            "public/client/TracyOverride.cpp",
            "public/client/TracyProfiler.cpp",
            "public/client/TracyRocprof.cpp",
            "public/client/TracySysPower.cpp",
            "public/client/TracySysTime.cpp",
            "public/client/TracySysTrace.cpp",
            "public/common/tracy_lz4.cpp",
            "public/common/TracyStackFrames.cpp",
            "public/common/TracySystem.cpp",
            "public/common/TracySocket.cpp",
        ],
    ),
    hdrs = [
        "public/tracy/Tracy.hpp",
        "public/tracy/TracyC.h",
        "public/tracy/TracyCUDA.hpp",
        "server/TracyWorker.hpp",
    ],
    copts = _COPTS,
    defines = [
        "TRACY_ENABLE",
        "TRACY_SAMPLING_HZ=2000",
        "TRACY_ON_DEMAND",
    ] + select({
        "@//:tracy_verbosity_1": ["TRACY_VERBOSITY=1"],
        "@//:tracy_verbosity_2": ["TRACY_VERBOSITY=2"],
        "//conditions:default": ["TRACY_VERBOSITY=1"],
    }) + select({
        "@//:tracy_rocm_enabled": ["TRACY_ROCPROF"],
        "//conditions:default": [],
    }),
    includes = [
        "public",
        # Server sources include common headers bare (e.g. "TracyProtocol.hpp");
        # upstream's server.cmake adds this directory the same way.
        "public/common",
        "server",
    ],
    linkopts = select({
        "@platforms//os:linux": ["-pthread"],
        "//conditions:default": [],  # macOS no extra linkopts
    }),
    output_name = "Tracy",
    tags = ["no-clang-tidy"],  # Third party code
    textual_hdrs = glob([
        "public/client/**/*.cpp",
        "public/common/**/*.cpp",
        "public/libbacktrace/**/*.cpp",
    ]),
    visibility = [
        "@//Kernels:__subpackages__",
        "@//Mojo:__subpackages__",
        "@//Support:__subpackages__",
        "@//utils/benchmarking/kepler:__subpackages__",
    ],
    deps = [
        "@capstone",
        "@llvm_zstd//:zstd",
        "@ppqsort",
    ] + select({
        # Only include CUPTI and the CUDA stubs when Tracy profiling is
        # explicitly enabled.  This avoids requiring libcupti.so / libcuda.so.1
        # / libcudart.so for regular builds.
        #
        # libcupti_stub provides all CUPTI symbols Tracy calls
        # directly via on-demand dlopen of libcupti.so.
        # libcuda_stub provides cuGetErrorString and cuStreamIsCapturing
        # (the two CUDA driver API symbols Tracy calls directly) via on-demand
        # dlopen of libcuda.so.1.
        # libcudart_stub provides cudaDeviceSynchronize (the single CUDA
        # runtime API symbol Tracy calls directly) via on-demand dlopen of
        # libcudart.so.
        # No stub carries a hard NEEDED entry for the underlying library.
        "@//:tracy_cuda_enabled": [
            "@//Support:libcuda_stub",
            "@//Support:libcudart_stub",
            "@//Support:libcupti_stub",
        ],
        "//conditions:default": [],
    }) + select({
        # librocprofiler_tracy_stub provides all rocprofiler-sdk symbols Tracy
        # calls directly via on-demand dlopen of librocprofiler-sdk.so.
        # The stub does not carry a hard NEEDED entry for the underlying library.
        "@//:tracy_rocm_enabled": [
            "@//Support:librocprofiler_tracy_stub",
            "@//bazel/public-patches:rocprofiler_headers",
        ],
        "//conditions:default": [],
    }),
)

# Upstream generates GitRef.hpp via CMake (cmake/GitRef.cmake); emit the
# no-git fallback since we build from a release tarball.
genrule(
    name = "gitref_hpp",
    outs = ["gitref/GitRef.hpp"],
    cmd = "printf '#pragma once\\nnamespace tracy { static inline const char* GitRef = \"unknown\"; }\\n' > $@",
)

cc_library(
    name = "gitref",
    hdrs = [":gitref_hpp"],
    includes = ["gitref"],
)

cc_binary(
    name = "capture_example",
    srcs = [
        "capture/src/CaptureOutput.cpp",
        "capture/src/CaptureOutput.hpp",
        "capture/src/capture.cpp",
    ],
    copts = _TOOL_COPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":Tracy",
        ":gitref",
    ],
)

cc_binary(
    name = "csvexport_example",
    srcs = ["csvexport/src/csvexport.cpp"],
    copts = _TOOL_COPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":Tracy",
        ":gitref",
    ],
)
