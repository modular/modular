"""Aliases for mojo packages."""

_STDLIB_PACKAGES = {
    "stdlib": "mojo/stdlib/stdlib",
    "python": "mojo/python/mojo",
    "test_utils": "mojo/stdlib/test/test_utils",
}

_PACKAGES = {
    "kv_cache": "max/kernels/src/kv_cache",
    "layout": "max/kernels/src/layout",
    "linalg": "max/kernels/src/linalg",
    "nn": "max/kernels/src/nn",
    "nvml": "max/kernels/src/nvml",
    "shmem": "max/kernels/src/shmem",
    "quantization": "max/kernels/src/quantization",
    "register": "max/kernels/src/register",
    "MOGGPrimitives": "max/kernels/src/Mogg/MOGGPrimitives",
    "MOGGKernelAPI": "max/kernels/src/Mogg/MOGGKernelAPI",
    "tensor": "max/kernels/src/extensibility/tensor",


    "compiler_internal": "max/kernels/src/extensibility/compiler_internal",
    "weights_registry": "max/kernels/src/weights_registry",
    "internal_utils": "max/kernels/src/internal_utils",
    "comm": "max/kernels/src/comm",
    "testdata": "max/kernels/test/testdata",
    "compiler": "max/compiler/src:compiler",
    "_cublas": "max/kernels/src/_cublas",
    "_cufft": "max/kernels/src/_cufft",
    "_curand": "max/kernels/src/_curand",
    "_cudnn": "max/kernels/src/_cudnn",
    "_rocblas": "max/kernels/src/_rocblas",
}

def _mojo_aliases_impl(rctx):
    alias_rules = []
    for name, target in _STDLIB_PACKAGES.items():
        alias_rules.append("""
alias(
    name = "{name}",
    actual = "@//{prefix}{target}",
)""".format(name = name, target = target, prefix = "{prefix}"))

    for name, target in _PACKAGES.items():
        alias_rules.append("""
alias(
    name = "{name}",
    actual = "@//{target}",
)""".format(name = name, target = target))

    build_content = """package(default_visibility = ["//visibility:public"])
{aliases}

""".format(aliases = "".join(alias_rules))

    rctx.file("BUILD.bazel", content = build_content.format(prefix = rctx.attr.prefix))
    rctx.file("mojo.bzl", content = """
ALL_MOJOPKGS = [
{packages}
]
""".format(packages = ",\n".join(['    "@mojo//:{}"'.format(name) for name in _PACKAGES.keys()])))

mojo_aliases = repository_rule(
    implementation = _mojo_aliases_impl,
    attrs = {
        "prefix": attr.string(
            doc = "The prefix of the modular/modular repo root",
            default = "",
        ),
    },
)
