"""Bazel toolchain representing the currently targeted GPU hardware"""

_NVIDIA_ASAN_OPTIONS = "use_sigaltstack=0,abort_on_error=1,disable_coredump=0,protect_shadow_gap=0"
_DEFAULT_ASAN_OPTIONS = "use_sigaltstack=0,abort_on_error=1,disable_coredump=0"

def _gpu_toolchain_alias_impl(ctx):
    toolchain = ctx.toolchains["@rules_mojo//:gpu_toolchain_type"]

    name = "__NO_GPU"
    brand = "NONE"
    gpu_cache_env = "__NO_GPU"
    if toolchain:
        toolchain_info = toolchain.mojo_gpu_toolchain_info
        name = toolchain_info.name
        brand = toolchain_info.brand
        gpu_cache_env = "{}-{}-{}".format(toolchain_info.name, toolchain_info.multi_gpu, toolchain_info.has_4_gpus)

    lit_prefix = brand.upper()
    if brand == "amdgpu":
        lit_prefix = "AMD"
    elif brand == "metal":
        lit_prefix = "APPLE"

    return [
        platform_common.TemplateVariableInfo({
            "GPU_ASAN_OPTIONS": _NVIDIA_ASAN_OPTIONS if brand == "nvidia" else _DEFAULT_ASAN_OPTIONS,
            "GPU_CACHE_ENV": gpu_cache_env,
            "GPU_LIT_FEATURE": "{}-GPU".format(name.upper()),
            "GPU_BRAND_LIT_FEATURE": "{}-GPU".format(lit_prefix),
        }),
    ]

gpu_toolchain_alias = rule(
    implementation = _gpu_toolchain_alias_impl,
    toolchains = [config_common.toolchain_type("@rules_mojo//:gpu_toolchain_type", mandatory = False)],
)
