"""Shared config that needs to be referenced by BUILD.bazel files but isn't important enough to be in api.bzl"""

# Used for linting unused targets, top level targets are potentially used
# externally, and therefore their deps are all considered used
TOP_LEVEL_TAG = "top-level"

# Used for linting unused targets, these targets might be unused and that's
# allowed, used sparingly, primarily used for macros that expand to multiple
# targets, some of which are optional
ALLOW_UNUSED_TAG = "maybe-unused"

# Default GPU memory for scheduling remote exec tests
DEFAULT_GPU_MEMORY = "0.8"

# `target_compatible_with` for tests whose graphs need an op registered only by
# the closed-source kernel packages (`internal_kernel_packages` in the internal
# MODULE.bazel). The source-only build doesn't have them, so the graph fails to
# compile there. Name the op at the use site so the test can come off this list
# once its kernel is public.
REQUIRES_INTERNAL_KERNELS = select({
    "//:oss_source_only_build": ["@platforms//:incompatible"],
    "//conditions:default": [],
})

MODULAR_CONFIGS = [
    "default",
    "debug_modular",
    "debug_everything",
    "dev",
    "ci_build",
    "release",
    "production",
    "asan",
    "tsan",
    "ubsan",
    "coverage",
]

ASYNCRT_PROFILING_LOCAL_DEFINES = [
    "MODULAR_ASYNCRT_MAX_PROFILING_LEVEL=0000000",
]

CONFIG_SECTION_LOCAL_DEFINES = [
    "MAX_CONFIG_SECTION=max",
    "MOJO_CONFIG_SECTION=mojo-max",
]

KERNEL_PROFILING_LOCAL_DEFINES = [
    "MODULAR_ENABLE_GPU_PROFILING=0",
    "MODULAR_ENABLE_GPU_PROFILING_DETAILED=0",
]
