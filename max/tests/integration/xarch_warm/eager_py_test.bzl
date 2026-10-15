"""`modular_eager_py_test`: `modular_py_test` + build-warmed eager-GC cache wiring.
"""

load("//bazel:api.bzl", "modular_py_test")

_WARM_PKG = "//max/tests/integration/xarch_warm"

# Imported lazily by the eager interpreter, so pydeps can't see it.
_INTERPRETER_OPS = "//max/python/max/_interpreter_ops"

# A warmed GPU lane gets its warm (CPU and accelerator slots) and a CPU lane the
# CPU-only warm. Any other GPU lane gets none: the interpreter then compiles its
# op families lazily, as without this macro, since a CPU-only warm would fail
# adoption on the first accelerator. Each pair of selects below is summed, so a
# lane takes the warmed-GPU half or the CPU half, never both.
_WARM_DATA = select({
    "//:b200_gpu": [_WARM_PKG + ":warm_gpu"],
    "//:mi355_gpu": [_WARM_PKG + ":warm_gpu"],
    "//conditions:default": [],
}) + select({
    "//:has_gpu": [],
    "//conditions:default": [_WARM_PKG + ":warm_cpu"],
})

# `XARCH_WARM_RLOCATIONS` must name the same target `_WARM_DATA` provides; the
# plugin resolves it into `MAX_EAGER_WARM_STORE`. `MAX_EAGER_OP_PRECOMPILE`
# loads the warm once at import rather than lazily per dispatch, and
# `MAX_EAGER_ALLOW_LAZY_COMPILE=0` makes an op model the warm lacks fail
# rather than compile. Without a warm, both would compile, so they come only
# with one.
_WARMED_ENV = {
    "MAX_EAGER_ALLOW_LAZY_COMPILE": "0",
    "MAX_EAGER_OP_PRECOMPILE": "1",
}
_GPU_WARM_ENV = _WARMED_ENV | {
    "XARCH_WARM_RLOCATIONS": "$(rlocationpaths {}:warm_gpu)".format(_WARM_PKG),
}
_CPU_WARM_ENV = _WARMED_ENV | {
    "XARCH_WARM_RLOCATIONS": "$(rlocationpaths {}:warm_cpu)".format(_WARM_PKG),
}
_WARM_ENV = select({
    "//:b200_gpu": _GPU_WARM_ENV,
    "//:mi355_gpu": _GPU_WARM_ENV,
    "//conditions:default": {},
}) | select({
    "//:has_gpu": {},
    "//conditions:default": _CPU_WARM_ENV,
})

# Interpreter only: a graph the interpreter refuses fails its test rather than
# compiling. A suite that may compile overrides it (`fallback-internal`); one
# test opts out with `@pytest.mark.compile_on_gpu(reason)`.
_EXECUTOR_ENV = {"MAX_EAGER_EXECUTOR": "interpreter"}

_HIDE_ACCELERATORS_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
}

# Warm-adoption pytest plugin.
_PLUGIN_LABEL = _WARM_PKG + ":eager_warm_plugin"
_PLUGIN_MODULE = "eager_warm_plugin"

def modular_eager_py_test(
        name,
        srcs,
        cpu_only = False,
        data = [],
        env = {},
        deps = [],
        ignore_extra_deps = [],
        args = [],
        **kwargs):
    """A `modular_py_test` that runs eager graphs interpreter-only.

    The test gets the interpreter's op families and runs with
    `MAX_EAGER_EXECUTOR=interpreter`. On a warmed lane every family loads its
    compiled sweep from the build-time warm store, and a warm that lacks a slot
    this test's devices need fails the test rather than compiling.

    Args:
        name: Test target name.
        srcs: Test sources.
        cpu_only: For tests that do not require a GPU. More of a hack to force
          the test to run with `CUDA_VISIBLE_DEVICES=""` and
          `HIP_VISIBLE_DEVICES=""` which allows the test to use cached ops.
        data: Extra runtime deps; merged with the warm store dirs.
        env: Extra env vars; merged over the warm and executor env (caller
            wins on conflict).
        deps: py deps; merged with the warm-adopt plugin lib and the
            interpreter's op families.
        ignore_extra_deps: Forwarded; the plugin lib and the op families are
            appended (neither is statically imported, so pydeps must not flag
            them).
        args: Extra pytest args; the `-p` plugin registration is prepended.
        **kwargs: Forwarded verbatim to `modular_py_test` (`gpu_constraints`,
            `size`, `tags`, `per_test_shard_count`, ...).
    """
    if cpu_only:
        warm_data = [_WARM_PKG + ":warm_cpu"]
        warm_env = _CPU_WARM_ENV | _HIDE_ACCELERATORS_ENV
    else:
        warm_data = _WARM_DATA
        warm_env = _WARM_ENV
    modular_py_test(
        name = name,
        srcs = srcs,
        data = data + warm_data,
        env = warm_env | _EXECUTOR_ENV | env,
        deps = deps + [_PLUGIN_LABEL, _INTERPRETER_OPS],
        ignore_extra_deps = ignore_extra_deps + [_PLUGIN_LABEL, _INTERPRETER_OPS],
        args = ["-p", _PLUGIN_MODULE] + args,
        **kwargs
    )
