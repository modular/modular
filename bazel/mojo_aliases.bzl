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
    actual = "@//max/kernels/src/kv_cache",
)

alias(
    name = "layout",
    actual = "@//max/kernels/src/layout",
)

alias(
    name = "linalg",
    actual = "@//max/kernels/src/linalg",
)

alias(
    name = "nn",
    actual = "@//max/kernels/src/nn",
)

alias(
    name = "nvml",
    actual = "@//max/kernels/src/nvml",
)

alias(
    name = "quantization",
    actual = "@//max/kernels/src/quantization",
)

alias(
    name = "register",
    actual = "@//max/kernels/src/register",
)

alias(
    name = "MOGGPrimitives",
    actual = "@//max/kernels/src/Mogg/MOGGPrimitives",
)

alias(
    name = "MOGGKernelAPI",
    actual = "@//max/kernels/src/Mogg/MOGGKernelAPI",
)

alias(
    name = "tensor_internal",
    actual = "@//max/kernels/src/extensibility/tensor_internal",
)

alias(
    name = "compiler_internal",
    actual = "@//max/kernels/src/extensibility/compiler_internal",
)
""")

mojo_aliases = repository_rule(
    implementation = _mojo_aliases_impl,
)
