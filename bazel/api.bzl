"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@com_github_grpc_grpc//bazel:python_rules.bzl", _py_grpc_library = "py_grpc_library")
load("@rules_cc//cc:cc_binary.bzl", _cc_binary = "cc_binary")
load("@rules_cc//cc:cc_library.bzl", _cc_library = "cc_library")
load("@rules_pkg//pkg:mappings.bzl", _strip_prefix = "strip_prefix")
load("@rules_proto//proto:defs.bzl", _proto_library = "proto_library")
load("//bazel/internal:lit.bzl", _lit_tests = "lit_tests")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_multi_py_version_test.bzl", _modular_multi_py_version_test = "modular_multi_py_version_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_binary.bzl", _modular_py_binary = "modular_py_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_library.bzl", _modular_py_library = "modular_py_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_test.bzl", _modular_py_test = "modular_py_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_venv.bzl", _modular_py_venv = "modular_py_venv")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_run_binary_test.bzl", _modular_run_binary_test = "modular_run_binary_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_binary.bzl", _mojo_binary = "mojo_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_filecheck_test.bzl", _mojo_filecheck_test = "mojo_filecheck_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_library.bzl", _mojo_library = "mojo_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test.bzl", _mojo_test = "mojo_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test_environment.bzl", _mojo_test_environment = "mojo_test_environment")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", _requirement = "pip_requirement")

modular_cc_binary = _cc_binary
modular_cc_library = _cc_library
modular_py_venv = _modular_py_venv
mojo_filecheck_test = _mojo_filecheck_test
mojo_test_environment = _mojo_test_environment
proto_library = _proto_library
py_grpc_library = _py_grpc_library
requirement = _requirement
strip_prefix = _strip_prefix

# Deps that aren't open source so we need to remap to pull from the wheel instead.
_DEPS_FROM_WHEEL = [
    "//max",
    "//max/_core_mojo",
    "//max/driver",
    "//max/dtype",
    "//max/engine",
    "//max/experimental",
    "//max/interfaces",
    "//max/mlir",
    "//max/profiler",
    "//max/support",
    "//max/torch",
    "//max:_core",
]

def _is_internal_reference(dep):
    """Check if a dependency is an internal reference."""
    return dep.startswith((
        "//GenericML",
        "//KGEN/",
        "//Kernels/",
        "//SDK/integration-test:",
        "//SDK/integration-test/pipelines/python",
        "//SDK:max",
    )) or "base_max_config_yaml_files" in dep or "benchmark_config_yaml_files" in dep

def _has_internal_reference(deps):
    return any([_is_internal_reference(dep) for dep in deps])

def _remove_internal_data(data):
    # TODO: This is a hack that potentially breaks things at runtime
    if type(data) != type([]):
        return []
    return [d for d in data if not _is_internal_reference(d)]

def _rewrite_deps(deps):
    """Rewrite dependencies to use the open-source package names, or to come from the wheel."""
    new_deps = []
    for dep in deps:
        if dep.startswith("//SDK/lib/API/python/tests/graph"):
            replaced_dep = dep.replace("//SDK/lib/API/python/tests/graph", "//tests/max/graph")
            new_deps.append(replaced_dep)
        elif dep.startswith("//SDK/lib/API/python/max/benchmark"):
            replaced_dep = dep.replace("//SDK/lib/API/python/max/benchmark", "//benchmark")
            new_deps.append(replaced_dep)
        elif dep.startswith("//SDK/lib/API/python/"):
            replaced_dep = dep.replace("//SDK/lib/API/python/", "//")
            if replaced_dep in _DEPS_FROM_WHEEL:
                replaced_dep = "@modular_wheel//:wheel"
            if replaced_dep not in new_deps:
                new_deps.append(replaced_dep)
        elif dep.startswith("//open-source/max/"):
            replaced_dep = dep.replace("//open-source/max/", "//")
            new_deps.append(replaced_dep)
        else:
            new_deps.append(dep)
    return new_deps

def _rewrite_trivial_env(env):
    if type(env) != type({}):
        return env
    new_env = {}
    for k, v in env.items():
        if v.startswith("SDK/lib/API/python/tests/graph"):
            new_env[k] = v.replace("SDK/lib/API/python/tests/graph", "tests/max/graph")
    return new_env

def modular_py_library(
        data = [],
        deps = [],
        visibility = ["//visibility:public"],
        **kwargs):
    if _has_internal_reference(deps):
        return

    _modular_py_library(
        data = _remove_internal_data(data),
        deps = _rewrite_deps(deps),
        visibility = visibility,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_py_binary(
        name,
        deps = [],
        data = [],
        env = {},
        **kwargs):
    if name == "pipelines":
        # TODO: Fix this hack, there is a layering issue with what is open source right now
        deps.append("//mojo/python/mojo")
        data = []
        env = {}

    # TODO: There is some data we can fix by pulling from the wheel
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return

    _modular_py_binary(
        name = name,
        data = data,
        env = env,
        deps = _rewrite_deps(deps),
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_py_test(
        name,
        deps = [],
        data = [],
        env = {},
        **kwargs):
    data = _rewrite_deps(data)
    deps = _rewrite_deps(deps)
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return

    _modular_py_test(
        name = name,
        data = data,
        env = _rewrite_trivial_env(env),
        deps = deps,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_multi_py_version_test(deps = [], data = [], env = {}, **kwargs):
    _modular_multi_py_version_test(
        deps = _rewrite_deps(deps),
        data = _rewrite_deps(data),
        env = _rewrite_trivial_env(env),
        **kwargs
    )

def mojo_library(deps = [], **kwargs):
    if _has_internal_reference(deps):
        return

    _mojo_library(
        deps = deps,
        **kwargs
    )

def mojo_binary(
        data = [],
        deps = [],
        **kwargs):
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return
    _mojo_binary(
        data = data,
        deps = deps,
        **kwargs
    )

def mojo_test(
        data = [],
        **kwargs):
    _mojo_test(
        data = _rewrite_deps(data),
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_run_binary_test(name, external_noop = False, **kwargs):
    if external_noop:
        return

    _modular_run_binary_test(
        name = name,
        **kwargs
    )

def lit_tests(tools = [], data = [], **kwargs):
    if _has_internal_reference(data) or _has_internal_reference(tools):
        return

    _lit_tests(
        data = data,
        tools = tools,
        **kwargs
    )

def _noop(**_kwargs):
    pass

mojo_kgen_lib = _noop
pkg_attributes = _noop
pkg_filegroup = _noop
pkg_files = _noop
