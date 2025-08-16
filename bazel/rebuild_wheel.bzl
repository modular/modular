"""Reassemble all the libs in the different wheels to live next to each other"""

_PLATFORM_MAPPINGS = {
    "linux_aarch64": "manylinux_2_34_aarch64",
    "linux_x86_64": "manylinux_2_34_x86_64",
    "macos_arm64": "macosx_13_0_arm64",
}

_WHEELS = [
    "max",
    "max_core",
    "mojo_compiler",
]

def _rebuild_wheel(rctx):
    for name in _WHEELS:
        rctx.download_and_extract(
            url = "{}/{}-{}-py3-none-{}.whl".format(
                rctx.attr.base_url,
                name,
                rctx.attr.version,
                _PLATFORM_MAPPINGS[rctx.attr.platform],
            ),
            type = "zip",
            strip_prefix = "{}-{}.data/platlib/".format(name, rctx.attr.version),
        )

    rctx.file(
        "BUILD.bazel",
        """
# print(glob(["max/**"]))
py_library(
    name = "max_core",
    data = glob([
        "max/_core.cpython-*",
        "modular/lib/**",
    ]),
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
