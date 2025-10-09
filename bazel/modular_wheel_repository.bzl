"""A repository rule for creating wheel accessors. Not enabled by default for compatibility with modular's internal repo."""

_PLATFORM_MAPPINGS = {
    "linux_aarch64": "manylinux_2_34_aarch64",
    "linux_x86_64": "manylinux_2_34_x86_64",
    "macos_arm64": "macosx_13_0_arm64",
}

_WHEELS = [
    "max_core",
    "mojo_compiler",
]

PYTHON_VERSIONS = [
    "310",
    "311",
    "312",
    "313",
]

def _rebuild_wheel(rctx):
    for py_version in PYTHON_VERSIONS:
        rctx.download_and_extract(
            url = "{base_url}/max-{version}-cp{py}-cp{py}-{platform}.whl".format(
                base_url = rctx.attr.base_url,
                version = rctx.attr.version,
                py = py_version,
                platform = _PLATFORM_MAPPINGS[rctx.attr.platform],
            ),
        )
    for name in _WHEELS:
        version_prefix = "0." if name.startswith("mojo") else ""
        version = version_prefix + rctx.attr.version
        rctx.download_and_extract(
            url = "{}/{}-{}-py3-none-{}.whl".format(
                rctx.attr.base_url,
                name,
                version,
                _PLATFORM_MAPPINGS[rctx.attr.platform],
            ),
            strip_prefix = "{}-{}.data/platlib/".format(name, version),
        )

    rctx.execute(["bash", "-c", "mv */platlib/max/_core.*.so max/"])
    rctx.execute(["mkdir", "-p", "max/_mlir/_mlir_libs"])
    rctx.execute(["bash", "-c", "mv */platlib/max/_mlir/_mlir_libs/_mlir.*.so max/_mlir/_mlir_libs/"])

    rctx.file(
        "BUILD.bazel",
        """
load("@rules_python//python:defs.bzl", "py_library")

# Subdirectories of the wheel that are part of this repo and therefore should
# be removed so that they're not accidentally used when testing changes that
# depend on some closed-source portions of the wheel.
_OPEN_SOURCE_GLOBS = [
    "modular/lib/mojo/*",
    "max/diagnostics/**",
    "max/entrypoints/**",
    "max/graph/**",
    "max/nn/**",
    "max/pipelines/**",
    "max/serve/**",
    "mojo/**",
]

py_library(
    name = "max",
    data = glob([
        "max/**",
        "modular/**",
    ], exclude = _OPEN_SOURCE_GLOBS),
    visibility = ["//visibility:public"],
    imports = ["."],
)""",
    )

rebuild_wheel = repository_rule(
    implementation = _rebuild_wheel,
    attrs = {
        "version": attr.string(
            mandatory = True,
        ),
        "platform": attr.string(
            values = _PLATFORM_MAPPINGS.keys(),
            mandatory = True,
        ),
        "base_url": attr.string(
            default = "https://dl.modular.com/public/nightly/python",
        ),
    },
)

def _modular_wheel_repository_impl(rctx):
    rctx.file("BUILD.bazel", """
load("@rules_pycross//pycross:defs.bzl", "pycross_wheel_library")
load("@@//bazel:api.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

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
    srcs = ["@@//bazel:mblack-main.py"],
    main = "@@//bazel:mblack-main.py",
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
