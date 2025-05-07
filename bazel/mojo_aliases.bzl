"""Aliases for mojo packages."""

def _mojo_aliases_impl(rctx):
    rctx.file("BUILD.bazel", content = """\
package(default_visibility = ["//visibility:public"])

alias(
    name = "stdlib",
    actual = "@//{prefix}mojo/stdlib/stdlib",
)

alias(
    name = "test_utils_srcs",
    actual = "@//{prefix}mojo/stdlib/test/test_utils:test_utils_srcs",
)

alias(
    name = "kv_cache",
    actual = "@//{prefix}max/kernels/src/kv_cache",
)

alias(
    name = "layout",
    actual = "@//{prefix}max/kernels/src/layout",
)

alias(
    name = "linalg",
    actual = "@//{prefix}max/kernels/src/linalg",
)

alias(
    name = "nn",
    actual = "@//{prefix}max/kernels/src/nn",
)

alias(
    name = "nvml",
    actual = "@//{prefix}max/kernels/src/nvml",
)

alias(
    name = "quantization",
    actual = "@//{prefix}max/kernels/src/quantization",
)

alias(
    name = "register",
    actual = "@//{prefix}max/kernels/src/register",
)

alias(
    name = "MOGGPrimitives",
    actual = "@//{prefix}max/kernels/src/Mogg/MOGGPrimitives",
)

alias(
    name = "MOGGKernelAPI",
    actual = "@//{prefix}max/kernels/src/Mogg/MOGGKernelAPI",
)

alias(
    name = "tensor_internal",
    actual = "@//{prefix}max/kernels/src/extensibility/tensor_internal",
)

alias(
    name = "compiler_internal",
    actual = "@//{prefix}max/kernels/src/extensibility/compiler_internal",
)

alias(
    name = "weights_registry",
    actual = "@//{prefix}max/kernels/src/weights_registry",
)

alias(
    name = "internal_utils",
    actual = "@//{prefix}max/kernels/src/internal_utils",
)

alias(
    name = "testdata",
    actual = "@//{prefix}max/kernels/test/testdata",
)

alias(
    name = "compiler",
    actual = "@//{prefix}max/compiler",
)
""".format(prefix = rctx.attr.prefix))

mojo_aliases = repository_rule(
    implementation = _mojo_aliases_impl,
    attrs = {
        "prefix": attr.string(
            doc = "The prefix of the modular/modular repo root",
            default = "",
        ),
    },
)
