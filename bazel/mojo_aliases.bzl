"""Aliases for mojo packages."""

def _mojo_aliases_impl(rctx):
    rctx.file("BUILD.bazel", content = """\
package(default_visibility = ["//visibility:public"])

alias(
    name = "stdlib",
    actual = "@//mojo/stdlib/stdlib",
)

alias(
    name = "kv_cache",
    actual = "@//mojo/kernels/src/kv_cache",
)

alias(
    name = "layout",
    actual = "@//mojo/kernels/src/layout",
)

alias(
    name = "linalg",
    actual = "@//mojo/kernels/src/linalg",
)

alias(
    name = "nn",
    actual = "@//mojo/kernels/src/nn",
)

alias(
    name = "nvml",
    actual = "@//mojo/kernels/src/nvml",
)

alias(
    name = "quantization",
    actual = "@//mojo/kernels/src/quantization",
)

alias(
    name = "register",
    actual = "@//mojo/kernels/src/register",
)

alias(
    name = "MOGGPrimitives",
    actual = "@//mojo/kernels/src/Mogg/MOGGPrimitives",
)

alias(
    name = "MOGGKernelAPI",
    actual = "@//mojo/kernels/src/Mogg/MOGGKernelAPI",
)
""")

mojo_aliases = repository_rule(
    implementation = _mojo_aliases_impl,
)
