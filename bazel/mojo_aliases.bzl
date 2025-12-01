"""Aliases for mojo packages."""

_PACKAGES = {
    "stdlib": "mojo/stdlib/stdlib",
    "python": "mojo/python/mojo",
    "test_utils": "mojo/stdlib/test/test_utils",
}

_MAX_PACKAGES = {
    "kv_cache": "kernels/src/kv_cache",
    "layout": "kernels/src/layout",
    "linalg": "kernels/src/linalg",
    "nn": "kernels/src/nn",
    "nvml": "kernels/src/nvml",
    "shmem": "kernels/src/shmem",
    "quantization": "kernels/src/quantization",
    "register": "kernels/src/register",
    "MOGGPrimitives": "kernels/src/Mogg/MOGGPrimitives",
    "MOGGKernelAPI": "kernels/src/Mogg/MOGGKernelAPI",
    "tensor": "kernels/src/extensibility/tensor",
    "compiler_internal": "kernels/src/extensibility/compiler_internal",
    "weights_registry": "kernels/src/weights_registry",
    "internal_utils": "kernels/src/internal_utils",
    "comm": "kernels/src/comm",
    "testdata": "kernels/test/testdata",
    "compiler": "compiler/src:compiler",
    "_cublas": "kernels/src/_cublas",
    "_cufft": "kernels/src/_cufft",
    "_curand": "kernels/src/_curand",
    "_cudnn": "kernels/src/_cudnn",
    "_rocblas": "kernels/src/_rocblas",
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
{max_packages}
]

def max_aliases():
    for name, target in {max_packages_dict}.items():
        native.alias(
            name = "{{name}}".format(name = name),
            actual = "//max/{{target}}".format(target = target),
            visibility = ["//visibility:public"],
        )
""".format(
        packages = "\n".join([
            '    "@mojo//:{}",'.format(name)
            for name in _PACKAGES.keys()
            if name != "python"
        ]),
        max_packages = "\n".join([
            '    "//max:{}",'.format(name)
            for name in _MAX_PACKAGES.keys()
        ]),
        max_packages_dict = _MAX_PACKAGES,
    ))

mojo_aliases = repository_rule(
    implementation = _mojo_aliases_impl,
    attrs = {
        "prefix": attr.string(
            doc = "The prefix of the modular/modular repo root",
            default = "",
        ),
    },
)
