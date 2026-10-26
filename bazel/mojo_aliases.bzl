"""Aliases for Mojo packages."""

_PACKAGES = {
    "std": "Mojo/stdlib/std",
    "python": "Mojo/python/mojo",
    "test_utils": "Mojo/stdlib/test/test_utils",
}

_EXTRA_ALIASES = {
    "__init__.mojo": "Mojo/stdlib/std:__init__.mojo",
    "std_srcs": "Mojo/stdlib/std:std_srcs",
}

_MAX_PACKAGES = {
    "algorithm": "kernels/src/algorithm",
    "kv_cache": "kernels/src/kv_cache",
    "layout": "kernels/src/layout",
    "linalg": "kernels/src/linalg",
    "nn": "kernels/src/nn",
    "profiling_range": "kernels/src/profiling_range",
    "shmem": "kernels/src/shmem",
    "quantization": "kernels/src/quantization",
    "extensibility": "kernels/src/graph_compiler/extensibility",
    "builtin_primitives": "kernels/src/graph_compiler/builtin_primitives",
    "builtin_kernels": "kernels/src/graph_compiler/builtin_kernels",
    "weights_registry": "kernels/src/weights_registry",
    "internal_utils": "kernels/src/internal_utils",
    "comm": "kernels/src/comm",
    "state_space": "kernels/src/state_space",
    "pipeline": "kernels/src/pipeline",
    "structured_kernels": "kernels/src/structured_kernels",
    "testdata": "kernels/test/testdata",
    "_cublas": "kernels/src/_cublas",
    "_cufft": "kernels/src/_cufft",
    "_cudnn": "kernels/src/_cudnn",
    "_rocblas": "kernels/src/_rocblas",
    "_miopen": "kernels/src/_miopen",
    "max_mojo": "mojo/max",
}

# Packages outside the `@mojo` aliases above that `ALL_MOJOPKGS` and
# `PROD_MOJOPKGS` include. Both build from source in the open-source tree.
#
# Kernel packages that exist only in the internal build go in the
# `internal_kernel_packages` attribute instead: a package listed there is
# simply absent from the open-source build, which keeps that build standalone.
INTERNAL_PACKAGES = []

# Packages that are marked testonly and cannot be used by production targets
_TESTONLY_MAX_PACKAGES = ["testdata"]

def _mojo_aliases_impl(rctx):
    alias_rules = []
    for name, target in (_PACKAGES | _EXTRA_ALIASES).items():
        override = rctx.attr.overrides.get(name)
        if override:
            actual = override
        else:
            actual = "\"@//" + target + "\""
        alias_rules.append("""
alias(
    name = "{name}",
    actual = {actual},
)""".format(name = name, actual = actual))

    build_content = """package(default_visibility = ["//visibility:public"])
{aliases}

""".format(aliases = "".join(alias_rules))

    rctx.file("BUILD.bazel", content = build_content)
    rctx.file("mojo.bzl", content = """
# Kernel packages that exist only in this build. Empty in the open-source
# build, where the kernels they register are not present at all -- so a BUILD
# file concatenates it instead of naming a `//Kernels` target, and reads the
# same in both trees. The graph compiler loads them if they resolve and skips
# them if they do not, so an empty list costs their ops and nothing else.
INTERNAL_KERNEL_MOJOPKGS = [
{internal_kernel_packages}
]

ALL_MOJOPKGS = [
{packages}
{max_packages}
{internal_packages}
] + INTERNAL_KERNEL_MOJOPKGS

# PROD_MOJOPKGS excludes testonly packages and can be used by non-test targets
PROD_MOJOPKGS = [
{prod_packages}
{prod_max_packages}
{internal_packages}
] + INTERNAL_KERNEL_MOJOPKGS

_OVERRIDES = {overrides}

def max_aliases():
    for name, target in {max_packages_dict}.items():
        override = _OVERRIDES.get(name)
        if override:
            actual = override
        else:
            actual = "//max/{{target}}".format(target = target)
        native.alias(
            name = "{{name}}".format(name = name),
            actual = actual,
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
        internal_packages = "\n".join([
            '    "{}",'.format(name)
            for name in INTERNAL_PACKAGES
        ]),
        internal_kernel_packages = "\n".join([
            '    "{}",'.format(name)
            for name in rctx.attr.internal_kernel_packages
        ]),
        prod_packages = "\n".join([
            '    "@mojo//:{}",'.format(name)
            for name in _PACKAGES.keys()
            if name not in ("python", "test_utils")
        ]),
        prod_max_packages = "\n".join([
            '    "//max:{}",'.format(name)
            for name in _MAX_PACKAGES.keys()
            if name not in _TESTONLY_MAX_PACKAGES
        ]),
        max_packages_dict = _MAX_PACKAGES,
        # Unescape quotes and strip leading/trailing quotes (needs to be in each select() element
        overrides = str(rctx.attr.overrides).replace("\"select({", "select({").replace("})\"", "})").replace("\\", ""),
    ))

mojo_aliases = repository_rule(
    implementation = _mojo_aliases_impl,
    attrs = {
        "internal_kernel_packages": attr.string_list(
            doc = "Kernel packages that exist only in this build, exposed as `INTERNAL_KERNEL_MOJOPKGS`.",
            default = [],
        ),
        "overrides": attr.string_dict(
            doc = "Override certain aliases to other targets.",
            default = {},
        ),
    },
)
