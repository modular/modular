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

# Subdirectories of the wheel that are part of this repo and therefore should
# be removed so that they're not accidentally used when testing changes that
# depend on some closed-source portions of the wheel.
# _OPEN_SOURCE_GLOBS = [
#     "*/platlib/modular/lib/mojo/*",
#     "max/entrypoints/*",
#     "max/graph/*",
#     "max/nn/*",
#     "max/pipelines/*",
#     "max/serve/*",
# ]

# [
#     py_library(
#         name = platform + "_wheel",
#         #install_exclude_globs = _OPEN_SOURCE_GLOBS + ["*/platlib/max/_core.cpython-*"],
#         tags = ["manual"],
#         #wheel = "@max_{}_wheel//file".format(platform),
#         deps = [
#             "@module_platlib_{}//:max_core".format(platform),
#         ],
#     )
#     for platform in [
#         "linux_x86_64",
#         "linux_aarch64",
#         "macos_arm64",
#     ]
# ]

pycross_wheel_library(
    name = "mblack-lib",
    tags = ["manual"],
    wheel = "@mblack__wheel//file",
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
