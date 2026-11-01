"""Shared constraints for the rust_kv tests."""

# Linux-only, and not under asan/ubsan: rules_rust doesn't stage
# `libclang_rt.<san>.so` into the test runfiles, so the binaries fail to load.
TEST_COMPATIBLE_WITH = ["@platforms//os:linux"] + select({
    "//:asan": ["@platforms//:incompatible"],
    "//:ubsan": ["@platforms//:incompatible"],
    "//conditions:default": [],
})

# For safe-Rust crates with nothing to sanitize: only skip under asan/ubsan,
# since the sanitizer runners cannot load the runtime for Rust test binaries.
NO_SANITIZER_COMPATIBLE_WITH = select({
    "//:asan": ["@platforms//:incompatible"],
    "//:ubsan": ["@platforms//:incompatible"],
    "//conditions:default": [],
})

# GPU (dlopen'd CUDA/ROCm) targets; gated to x86_64 Linux like their consumers.
GPU_PLATFORM = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
]
