"""Aliases for mojo packages."""

_PACKAGES = {
    "stdlib": "mojo/stdlib/stdlib",
    "test_utils": "mojo/stdlib/test/test_utils",
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
    "tensor_internal": "max/kernels/src/extensibility/tensor_internal",
    "compiler_internal": "max/kernels/src/extensibility/compiler_internal",
    "weights_registry": "max/kernels/src/weights_registry",
    "internal_utils": "max/kernels/src/internal_utils",
    "comm": "max/kernels/src/comm",
    "testdata": "max/kernels/test/testdata",
    "compiler": "max/compiler/src:compiler",
}

def _mojo_aliases_impl(rctx):
    alias_rules = []
    for name, target in _PACKAGES.items():
        alias_rules.append("""
alias(
    name = "{name}",
    actual = "@//{prefix}{target}",
)""".format(name = name, target = target, prefix = "{prefix}"))

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
