load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_hip_version",
    srcs = [
        "projects/hip/VERSION",
        "@//bazel/public-patches:hip_version.h.in",
    ],
    outs = ["generated/hip/hip_version.h"],
    cmd = "MAJOR=$$(sed -n '2p' $(location projects/hip/VERSION)) && " +
          "MINOR=$$(sed -n '4p' $(location projects/hip/VERSION)) && " +
          "PATCH=$$(sed -n '6p' $(location projects/hip/VERSION)) && " +
          "sed \"s/@MAJOR@/$$MAJOR/;s/@MINOR@/$$MINOR/;s/@PATCH@/$$PATCH/\" " +
          "$(location @//bazel/public-patches:hip_version.h.in) > $@",
)

cc_library(
    name = "hip_headers",
    hdrs = glob([
        "projects/hip/include/hip/**/*.h",
        "projects/clr/hipamd/include/hip/**/*.h",
        "projects/clr/hipamd/include/hip/**/*.hpp",
    ]) + [":gen_hip_version"],
    defines = ["__HIP_PLATFORM_AMD__"],
    includes = [
        "generated",
        "projects/clr/hipamd/include",
        "projects/hip/include",
    ],
)

# ---------------------------------------------------------------------------
# HSA runtime headers
# ---------------------------------------------------------------------------
# Source files are at projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h etc.
# Code includes them as <hsa/hsa.h>, so we strip the source prefix and add
# an "hsa" include prefix.
cc_library(
    name = "hsa_headers",
    hdrs = glob(
        ["projects/rocr-runtime/runtime/hsa-runtime/inc/**/*.h"],
        allow_empty = True,
    ),
    include_prefix = "hsa",
    strip_include_prefix = "projects/rocr-runtime/runtime/hsa-runtime/inc",
)

# ---------------------------------------------------------------------------
# HSAKMT headers
# ---------------------------------------------------------------------------
# Source tree already has hsakmt/ subdirectory, so we just point the include
# path at the parent directory and <hsakmt/hsakmttypes.h> resolves directly.
cc_library(
    name = "hsakmt_headers",
    hdrs = glob(
        ["projects/rocr-runtime/libhsakmt/include/hsakmt/**/*.h"],
        allow_empty = True,
    ),
    includes = ["projects/rocr-runtime/libhsakmt/include"],
)

# ---------------------------------------------------------------------------
# rocprofiler-sdk generated version headers
# ---------------------------------------------------------------------------
# rocprofiler-sdk ships version.h.in and defines.h.in that use CMake @VAR@
# substitution. We generate them with sed. The version tracks therock-7.11
# which ships rocprofiler-sdk 1.1.0.

# Common sed substitutions shared by the version header genrules.
# These fields are version metadata only; use generic values since the
# headers are compiled for both x86_64 and aarch64.
_ROCPROF_VERSION_SUBS = (
    "-e 's/@PROJECT_VERSION_MAJOR@/1/g' " +
    "-e 's/@PROJECT_VERSION_MINOR@/1/g' " +
    "-e 's/@PROJECT_VERSION_PATCH@/0/g' " +
    "-e 's/@FULL_VERSION_STRING@/1.1.0/g' " +
    "-e 's/@ROCPROFILER_SDK_GIT_DESCRIBE@/therock-7.11/g' " +
    "-e 's/@ROCPROFILER_SDK_GIT_REVISION@/unknown/g' "
)

_ROCPROF_CMAKE_SUBS = (
    "-e 's/@CMAKE_LIBRARY_ARCHITECTURE@/unknown/g' " +
    "-e 's/@CMAKE_SYSTEM_NAME@/Linux/g' " +
    "-e 's/@CMAKE_SYSTEM_PROCESSOR@/unknown/g' " +
    "-e 's/@CMAKE_SYSTEM_VERSION@/unknown/g' " +
    "-e 's/@CMAKE_CXX_COMPILER_ID@/Clang/g' " +
    "-e 's/@CMAKE_CXX_COMPILER_VERSION@/unknown/g' "
)

genrule(
    name = "gen_rocprofiler_version",
    srcs = ["projects/rocprofiler-sdk/source/include/rocprofiler-sdk/version.h.in"],
    outs = ["generated/rocprofiler-sdk/version.h"],
    cmd = "sed " + _ROCPROF_VERSION_SUBS + _ROCPROF_CMAKE_SUBS + "$(SRCS) > $@",
)

genrule(
    name = "gen_rocprofiler_ext_version",
    srcs = ["projects/rocprofiler-sdk/source/include/rocprofiler-sdk/ext_version.h.in"],
    outs = ["generated/rocprofiler-sdk/ext_version.h"],
    cmd = "sed " +
          "-e 's/@HSA_RUNTIME_VERSION_MAJOR@/1/g' " +
          "-e 's/@HSA_RUNTIME_VERSION_MINOR@/16/g' " +
          "-e 's/#cmakedefine \\(ROCPROFILER_[A-Z_]*\\) @\\(.*\\)@/#define \\1 0/g' " +
          "$(SRCS) > $@",
)

genrule(
    name = "gen_roctx_version",
    srcs = ["projects/rocprofiler-sdk/source/include/rocprofiler-sdk-roctx/version.h.in"],
    outs = ["generated/rocprofiler-sdk-roctx/version.h"],
    cmd = "sed " + _ROCPROF_VERSION_SUBS + _ROCPROF_CMAKE_SUBS + "$(SRCS) > $@",
)

# ---------------------------------------------------------------------------
# rocprofiler-sdk headers
# ---------------------------------------------------------------------------
# Exposes all rocprofiler-sdk public headers with their transitive deps (HSA,
# HSAKMT, HIP). Tracy's TracyRocprof.cpp includes
# <rocprofiler-sdk/registration.h> and <rocprofiler-sdk/rocprofiler.h>.
# rocprofiler-sdk/marker/api_args.h transitively includes
# <rocprofiler-sdk-roctx/api_trace.h>, so we also expose the roctx headers.
cc_library(
    name = "rocprofiler_headers",
    hdrs = glob(
        [
            "projects/rocprofiler-sdk/source/include/rocprofiler-sdk/**/*.h",
            "projects/rocprofiler-sdk/source/include/rocprofiler-sdk/**/*.hpp",
            "projects/rocprofiler-sdk/source/include/rocprofiler-sdk-roctx/**/*.h",
        ],
        allow_empty = True,
    ) + [
        ":gen_rocprofiler_ext_version",
        ":gen_rocprofiler_version",
        ":gen_roctx_version",
    ],
    defines = ["__HIP_PLATFORM_AMD__"],
    includes = [
        "generated",
        "projects/rocprofiler-sdk/source/include",
    ],
    deps = [
        ":hip_headers",
        ":hsa_headers",
        ":hsakmt_headers",
    ],
)
