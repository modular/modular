"""A repository rule for creating wheel accessors. Not enabled by default for compatibility with modular's internal repo."""

def _modular_wheel_repository_impl(rctx):
    rctx.file("BUILD.bazel", """
load("@rules_pycross//pycross:defs.bzl", "pycross_wheel_library")
load("@@//bazel:api.bzl", "requirement")

alias(
    name = "wheel",
    actual = select({
        "@//:linux_aarch64": "@module_platlib_linux_aarch64//:max",
        "@//:linux_x86_64": "@module_platlib_linux_x86_64//:max",
        "@platforms//os:macos": "@module_platlib_macos_arm64//:max",
    }),
    visibility = ["//visibility:public"],
)

pycross_wheel_library(
    name = "mblack-lib",
    tags = ["manual"],
    wheel = "@mblack_wheel//file",
)

py_binary(
    name = "mblack",
    srcs = ["@@//bazel/lint:mblack-wrapper.py"],
    main = "@@//bazel/lint:mblack-wrapper.py",
    visibility = ["//visibility:public"],
    deps = [
        ":mblack-lib",
        requirement("click"),
        requirement("mypy-extensions"),
        requirement("pathspec"),
        requirement("platformdirs"),
        requirement("tomli"),
    ],
)
""")

modular_wheel_repository = repository_rule(
    implementation = _modular_wheel_repository_impl,
)
