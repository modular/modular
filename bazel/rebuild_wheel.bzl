"""Reassemble all the libs in the different wheels to live next to each other"""

_PLATFORM_MAPPINGS = {
    "linux_aarch64": "manylinux_2_34_aarch64",
    "linux_x86_64": "manylinux_2_34_x86_64",
    "macos_arm64": "macosx_13_0_arm64",
}

_WHEELS = [
    ("max", False),
    # Hack: we 'download' twice and strip the prefix on the
    # second one to get the bindings in the right place.
    # There's definitely a better way to do this.
    ("max", True),
    ("max_core", True),
    ("mojo_compiler", True),
]

def _rebuild_wheel(rctx):
    for name, strip in _WHEELS:
        strip_prefix = "{}-{}.data/platlib/".format(name, rctx.attr.version) if strip else ""
        rctx.download_and_extract(
            url = "{}/{}-{}-py3-none-{}.whl".format(
                rctx.attr.base_url,
                name,
                rctx.attr.version,
                _PLATFORM_MAPPINGS[rctx.attr.platform],
            ),
            strip_prefix = strip_prefix,
        )

    rctx.file(
        "BUILD.bazel",
        """
# Subdirectories of the wheel that are part of this repo and therefore should
# be removed so that they're not accidentally used when testing changes that
# depend on some closed-source portions of the wheel.
_OPEN_SOURCE_GLOBS = [
    "modular/lib/mojo/*",
    "max/entrypoints/*",
    "max/graph/*",
    "max/nn/*",
    "max/pipelines/*",
    "max/serve/*",
]

py_library(
    name = "max",
    data = glob([
        "max/**",
        "modular/lib/**",
        "modular/bin/**",
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
