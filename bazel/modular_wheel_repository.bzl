"""A repository rule for creating wheel accessors. Not enabled by default for compatibility with modular's internal repo."""

def _symlink_bindings(ctx):
    # Strip the leading max-<version>.data/platlib from bindings files to make all files in the archive have a single import

    symlinks = []
    for i in range(len(ctx.attr.files)):
        file = ctx.files.files[i]
        string = ctx.attr.file_strings[i]
        output = string.split("/", 2)[2]
        output_file = ctx.actions.declare_file(output)
        ctx.actions.symlink(
            output = output_file,
            target_file = file,
        )
        symlinks.append(output_file)

    return DefaultInfo(files = depset(symlinks))

symlink_bindings = rule(
    implementation = _symlink_bindings,
    attrs = {
        "files": attr.label_list(allow_files = True),
        "file_strings": attr.string_list(),
    },
)

_PLATFORM_MAPPINGS = {
    "linux_aarch64": "manylinux_2_34_aarch64",
    "linux_x86_64": "manylinux_2_34_x86_64",
    "macos_arm64": "macosx_13_0_arm64",
}

def _rebuild_wheel(rctx):
    rctx.download_and_extract(
        url = "{}/max-{}-py3-none-{}.whl".format(
            rctx.attr.base_url,
            rctx.attr.version,
            _PLATFORM_MAPPINGS[rctx.attr.platform],
        ),
    )

    rctx.file(
        "BUILD.bazel",
        """
# Subdirectories of the wheel that are part of this repo and therefore should
# be removed so that they're not accidentally used when testing changes that
# depend on some closed-source portions of the wheel.
_OPEN_SOURCE_GLOBS = [
    "*/platlib/max/lib/mojo/*",
    "max/entrypoints/**",
    "max/graph/**",
    "max/nn/**",
    "max/pipelines/**",
    "max/serve/**",
    "mojo/**",
]

load("@@//bazel:modular_wheel_repository.bzl", "symlink_bindings")

symlink_bindings(
    name = "symlinks",
    files = glob(["*/platlib/**"]),
    file_strings = glob(["*/platlib/**"]),
)

py_library(
    name = "max",
    data = glob([
        "max/**",
        "*/platlib/max/**",
    ], exclude = _OPEN_SOURCE_GLOBS) + [
        ":symlinks",
    ],
    visibility = ["//visibility:public"],
    imports = ["."],
)""".format(rctx.attr.version),
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
