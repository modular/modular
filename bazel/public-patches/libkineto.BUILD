# Bazel build for libkineto.
#
# Pinned to the SHA bundled with torch==2.12.0+cu130 (April 2026), the
# pairing empirically validated against HTA 0.5.0 in the initial spike.
#
# ----------------------------------------------------------------------
# Cross-cutting build fixes maintained by this file (revisit on SHA bump):
#
#   1. Source list comes from libkineto's own libkineto_defs.bzl — plain
#      Starlark, load()-able directly. The only Bazel-specific tweak
#      needed is deduping src/Demangle.cpp (CMake silently dedupes; Bazel
#      does not).
#   2. -fexceptions + -frtti are added to LIBKINETO_COPTS because the rest
#      of the MAX codebase compiles with -fno-exceptions / -fno-rtti and
#      libkineto needs both. The flags are scoped via `copts` so they do
#      not leak to dependents.
#   3. CUDA's cuda_runtime.h includes "crt/host_config.h", which the
#      cuda_x86_64 / cuda_sbsa redistributables don't ship. No special
#      handling is needed here: cuda_runtime_headers (cuda_cudart.BUILD's
#      crt_host_config_h genrule) already synthesizes an empty stub on the
#      include path, so depending on :cuda_runtime_headers is sufficient.
#   4. ENABLE_IPC_FABRIC is defined so libkineto's IpcFabricConfigClient /
#      DaemonConfigLoader connect to a Dynolog daemon for on-demand captures.
#      The required ipcfabric headers come from @dynolog//:ipcfabric_headers
#      (see dynolog.BUILD for why the kineto tarball can't supply them).
#      Verified end-to-end against Dynolog 0.6.0.
#
# Deferred (intentionally gated off in v1):
#
#   - Linker version script (libkineto.lds) for symbol scoping. Only
#     relevant once libkineto is linked into a shared library;
#     today the cc_library produces a static archive, so symbol scoping
#     happens at the final .so link, not here.
#   - aarch64 Linux. The CUPTI build + CUDA dep chain are validated only
#     on x86_64, so target_compatible_with conservatively restricts to it.
#     (The CUDA header chain itself resolves on aarch64 via cuda_sbsa, so
#     lifting this is mostly a validation exercise.)

load(
    "@libkineto//:libkineto/libkineto_defs.bzl",
    "KINETO_COMPILER_FLAGS",
    "get_libkineto_api_srcs",
    "get_libkineto_cupti_srcs",
    "get_libkineto_rocprofiler_srcs",
)
load("@rules_cc//cc:defs.bzl", "cc_library")

# Restrict to known consumers rather than //visibility:public.  Today only
# //max/internal/profiler (the MAXProfilerPlugin) links against
# libkineto.  Pinning visibility here prevents accidental future consumers
# from picking up a transitive dependency on libkineto + CUPTI without an
# explicit review.  Add new packages here as they need access.
package(default_visibility = [
    "@//max/internal/profiler:__pkg__",
])

# Public API headers. Consumers that only need the API surface (and don't
# need to link against CUPTI) can depend on this directly.
cc_library(
    name = "libkineto_headers",
    hdrs = glob(["libkineto/include/**/*.h"]),
    includes = ["libkineto/include"],
)

# Internal libkineto source headers. Exposed as a separate target so the
# `includes` attribute can put `libkineto/src` on the compile search path
# without leaking those headers as part of the public API. Visibility is
# private: several of these headers reference `KINETO_NAMESPACE`, which
# only `:libkineto` / `:libkineto_api_objs` define (via local_defines).
# Consumers outside this BUILD file should depend on `:libkineto_headers`
# or `:libkineto`, not this target.
cc_library(
    name = "libkineto_src_headers",
    hdrs = glob(["libkineto/src/**/*.h"]),
    includes = ["libkineto/src"],
    visibility = ["//visibility:private"],
    deps = [":libkineto_headers"],
)

# Upstream libkineto exports KINETO_COMPILER_FLAGS from libkineto_defs.bzl:
# ["-fexceptions", "-Wno-deprecated-declarations", "-Wno-unused-function",
#  "-Wno-unused-private-field"]. Reusing it keeps us in sync across SHA
# bumps. We add -frtti on top because libkineto uses dynamic_cast in
# CuptiRangeProfilerConfig.h and the MAX toolchain default is -fno-rtti.
# Both -fexceptions and -frtti are scoped via `copts` so they apply to
# libkineto's own sources but do not leak to dependents (the rest of the
# MAX codebase compiles with -fno-exceptions / -fno-rtti).
LIBKINETO_COPTS = KINETO_COMPILER_FLAGS + ["-frtti"]

# Defines that only libkineto's own sources need. `local_defines` means
# they apply during compilation of this cc_library's `srcs` but do not
# propagate to consumers — libkineto sources use these for internal
# behavior (namespace, CUPTI gating) and don't need to be visible
# downstream.
LIBKINETO_LOCAL_DEFINES = [
    # Required: namespace under which libkineto's symbols are defined.
    "KINETO_NAMESPACE=libkineto",
    # Enables the CUPTI activity backend. The initial spike confirmed this
    # works against torch 2.12.0+cu130's bundled libkineto at the same
    # SHA, so we ship CUPTI as the v1 backend.
    "HAS_CUPTI=1",
]

# libkineto's GPU backends are compile-time mutually exclusive (the #if
# HAS_CUPTI / #elif HAS_ROCTRACER chain in ActivityProfilerController.cpp
# does no runtime vendor detection), so the AMD flavor is a sibling
# cc_library rather than extra defines on :libkineto. HAS_ROCTRACER is the
# single umbrella define for both AMD paths at this SHA; leaving
# ROCTRACER_FALLBACK undefined selects the rocprofiler-sdk implementation
# (what upstream CMake auto-picks for ROCm >= 6.4) instead of the deprecated
# legacy-roctracer one. __HIP_PLATFORM_AMD__ propagates from the
# rocprofiler/HIP headers dep, so it is not repeated here.
LIBKINETO_ROCM_LOCAL_DEFINES = [
    "KINETO_NAMESPACE=libkineto",
    "HAS_ROCTRACER=1",
]

# The shared API objects (libkineto_api.cpp, ThreadUtil.cpp). Split out
# because libkineto's CMake build keeps these separate from the GPU
# backend so the API surface stays linkable even when no GPU is detected.
# Same platform constraint as `:libkineto` so the constraint lives with
# the sources rather than relying on transitive enforcement through the
# `:libkineto` consumer.
cc_library(
    name = "libkineto_api_objs",
    srcs = [
        "libkineto/" + src
        for src in get_libkineto_api_srcs()
    ],
    copts = LIBKINETO_COPTS,
    local_defines = LIBKINETO_LOCAL_DEFINES,
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    deps = [
        ":libkineto_headers",
        ":libkineto_src_headers",
        "@fmt",
    ],
)

# Upstream libkineto_defs.bzl lists src/Demangle.cpp in both the cupti
# source list and the cpu-only source list. CMake's add_library dedupes
# silently; Bazel's srcs attribute does not. Starlark lacks set literals,
# so we use a dict whose keys are the dedup set, then sort for stable
# action keys across builds.
_LIBKINETO_CUPTI_SRCS = sorted({
    "libkineto/" + src: None
    for src in get_libkineto_cupti_srcs(with_api = False)
}.keys())

# Full libkineto with the CUPTI backend wired in.
cc_library(
    name = "libkineto",
    srcs = _LIBKINETO_CUPTI_SRCS,
    copts = LIBKINETO_COPTS,
    # ENABLE_IPC_FABRIC is scoped to this target (not the shared
    # LIBKINETO_LOCAL_DEFINES) because only the cupti src set compiles the IPC
    # sources (IpcFabricConfigClient.cpp / DaemonConfigLoader.cpp). Keeping the
    # define beside the @dynolog//:ipcfabric_headers dep they require avoids a
    # define-without-headers skew if a future SHA bump moves those sources.
    local_defines = LIBKINETO_LOCAL_DEFINES + ["ENABLE_IPC_FABRIC=1"],
    # The CUPTI build is validated only on x86_64 today, so restrict to it
    # conservatively. aarch64 + AMD-only builds keep the headers-only target
    # ":libkineto_headers" instead.
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    deps = [
        ":libkineto_api_objs",
        ":libkineto_headers",
        ":libkineto_src_headers",
        # cuda_cupti_headers provides <cupti.h> + transitively
        # cuda_runtime_api.h. Same target Tracy CUDA depends on. The
        # `@//` prefix reaches back into the main MAX workspace from
        # this externally-loaded BUILD file. cuda_runtime_headers also
        # supplies the synthesized crt/host_config.h (cuda_cudart.BUILD's
        # crt_host_config_h genrule) that the redistributables don't ship,
        # so the CUDA header chain resolves without a separate shim.
        "@//bazel/public-patches:cuda_cupti_headers",
        "@//bazel/public-patches:cuda_runtime_headers",
        "@dynolog//:ipcfabric_headers",
        "@fmt",
    ],
)

# Sorted-set shape mirrors _LIBKINETO_CUPTI_SRCS above. At this SHA the
# rocprofiler list has no duplicate (only the cupti list carries Demangle.cpp
# twice), so the dedup just guards future SHA bumps.
_LIBKINETO_ROCM_SRCS = sorted({
    "libkineto/" + src: None
    for src in get_libkineto_rocprofiler_srcs(with_api = False)
}.keys())

# Full libkineto with the rocprofiler-sdk (AMD) backend wired in. Selected by
# //max/internal/profiler:MAXProfilerPlugin under the //:amd_gpu config in
# place of :libkineto — never link both into one binary (each registers its
# own ActivityProfilerProxy).
#
# :libkineto_api_objs is shared with the CUPTI flavor. Its HAS_CUPTI define
# is inert there — neither libkineto_api.cpp nor ThreadUtil.cpp references
# any vendor API — and sharing keeps the two flavors from compiling the API
# objects twice. Revisit on a SHA bump if those files grow vendor #ifdefs.
cc_library(
    name = "libkineto_rocm",
    srcs = _LIBKINETO_ROCM_SRCS,
    copts = LIBKINETO_COPTS,
    # ENABLE_IPC_FABRIC scoped here for the same reason as :libkineto: the
    # cpu-only source set carries IpcFabricConfigClient.cpp /
    # DaemonConfigLoader.cpp, which need @dynolog//:ipcfabric_headers.
    local_defines = LIBKINETO_ROCM_LOCAL_DEFINES + ["ENABLE_IPC_FABRIC=1"],
    # AMD GPU hosts are Linux x86_64 (the MI355 CI executors); keep the same
    # conservative constraint as the CUPTI flavor.
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    deps = [
        ":libkineto_api_objs",
        ":libkineto_headers",
        ":libkineto_src_headers",
        # rocprofiler_headers carries the rocprofiler-sdk public headers plus
        # transitively the HIP / HSA headers (and the __HIP_PLATFORM_AMD__
        # define) that DeviceUtil.h / RocLogger.h / RocprofActivity.h need.
        # The runtime libraries are NOT linked: the plugin links the
        # on-demand dlopen stubs (librocprofiler_stub_static /
        # libamdhip64_stub_static) instead, so no vendor DT_NEEDED entries.
        "@//bazel/public-patches:rocprofiler_headers",
        "@dynolog//:ipcfabric_headers",
        "@fmt",
    ],
)
