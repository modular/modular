"""A repository rule for creating wheel accessors. Not enabled by default for compatibility with modular's internal repo."""

def _modular_wheel_repository_impl(rctx):
    rctx.file("BUILD.bazel", """
load("@rules_pycross//pycross:defs.bzl", "pycross_wheel_library")

alias(
    name = "wheel",
    actual = select({
        "@//:linux_aarch64": ":linux_aarch64_wheel",
        "@//:linux_x86_64": ":linux_x86_64_wheel",
        "@platforms//os:macos": ":macos_arm64_wheel",
    }),
    visibility = ["//visibility:public"],
)

# Subdirectories of the wheel that are part of this repo and therefore should
# be removed so that they're not accidentally used when testing changes that
# depend on some closed-source portions of the wheel.
_OPEN_SOURCE_GLOBS = [
    "*/platlib/max/lib/mojo/*",
    "max/entrypoints/*",
    "max/nn/*",
    "max/pipelines/*",
    "max/serve/*",
]

pycross_wheel_library(
    name = "linux_x86_64_wheel",
    install_exclude_globs = _OPEN_SOURCE_GLOBS,
    tags = ["manual"],
    wheel = "@modular_linux_x86_64//file",
)

pycross_wheel_library(
    name = "linux_aarch64_wheel",
    install_exclude_globs = _OPEN_SOURCE_GLOBS,
    tags = ["manual"],
    wheel = "@modular_linux_aarch64//file",
)

pycross_wheel_library(
    name = "macos_arm64_wheel",
    install_exclude_globs = _OPEN_SOURCE_GLOBS,
    tags = ["manual"],
    wheel = "@modular_macos_arm64//file",
)
""")

modular_wheel_repository = repository_rule(
    implementation = _modular_wheel_repository_impl,
)
