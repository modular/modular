"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@rules_pkg//pkg:mappings.bzl", _pkg_filegroup = "pkg_filegroup", _pkg_files = "pkg_files", _strip_prefix = "strip_prefix")
load("//bazel/internal:copy_files.bzl", _copy_files = "copy_files")  # buildifier: disable=bzl-visibility
load("//bazel/internal:dialect_checksum.bzl", _dialect_checksum = "dialect_checksum")  # buildifier: disable=bzl-visibility
load("//bazel/internal:kgen.bzl", _kgen_kernel = "kgen_kernel")  # buildifier: disable=bzl-visibility
load("//bazel/internal:lit.bzl", _lit_tests = "lit_tests")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mef.bzl", _mef = "mef")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_binary.bzl", _modular_cc_binary = "modular_cc_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_library.bzl", _modular_cc_library = "modular_cc_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_test.bzl", _modular_cc_test = "modular_cc_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_generate_stubfiles.bzl", _modular_generate_stubfiles = "modular_generate_stubfiles")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_genrule.bzl", _modular_genrule = "modular_genrule")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_multi_py_version_test.bzl", _modular_multi_py_version_test = "modular_multi_py_version_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_nanobind_extension.bzl", _modular_nanobind_extension = "modular_nanobind_extension")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_nanobind_library.bzl", _modular_nanobind_library = "modular_nanobind_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_binary.bzl", _modular_py_binary = "modular_py_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_library.bzl", _modular_py_library = "modular_py_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_test.bzl", _modular_py_test = "modular_py_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_venv.bzl", _modular_py_venv = "modular_py_venv")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_pyo3_extension.bzl", _modular_pyo3_extension = "modular_pyo3_extension")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_run_binary_test.bzl", _modular_run_binary_test = "modular_run_binary_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_shared_library.bzl", _modular_shared_library = "modular_shared_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_sphinx_docs.bzl", _modular_sphinx_docs = "modular_sphinx_docs")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_versioned_expand_template.bzl", _modular_versioned_expand_template = "modular_versioned_expand_template")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_binary.bzl", _mojo_binary = "mojo_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_filecheck_test.bzl", _mojo_filecheck_test = "mojo_filecheck_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_library.bzl", _mojo_library = "mojo_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_merged_precompile.bzl", _mojo_merged_precompile = "mojo_merged_precompile")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_shared_library.bzl", _mojo_shared_library = "mojo_shared_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test.bzl", _mojo_test = "mojo_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test_environment.bzl", _mojo_test_environment = "mojo_test_environment")  # buildifier: disable=bzl-visibility
load("//bazel/internal:overlay_layer.bzl", _overlay_layer = "overlay_layer")  # buildifier: disable=bzl-visibility
load("//bazel/internal:overlay_srcs.bzl", _overlay_srcs = "overlay_srcs")  # buildifier: disable=bzl-visibility
load("//bazel/internal:py_repl.bzl", _py_repl = "py_repl")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", _requirement = "pip_requirement")

dialect_checksum = _dialect_checksum
kgen_kernel = _kgen_kernel
lit_tests = _lit_tests
modular_genrule = _modular_genrule
modular_multi_py_version_test = _modular_multi_py_version_test
modular_py_library = _modular_py_library
modular_py_venv = _modular_py_venv
modular_run_binary_test = _modular_run_binary_test
modular_sphinx_docs = _modular_sphinx_docs
modular_versioned_expand_template = _modular_versioned_expand_template
mojo_filecheck_test = _mojo_filecheck_test
mojo_merged_precompile = _mojo_merged_precompile
mojo_test = _mojo_test
mojo_test_environment = _mojo_test_environment
overlay_layer = _overlay_layer
overlay_srcs = _overlay_srcs
pkg_filegroup = _pkg_filegroup
pkg_files = _pkg_files
py_repl = _py_repl
requirement = _requirement
strip_prefix = _strip_prefix

def modular_py_test(tags = [], **kwargs):
    if "external-exclusive" in tags:
        tags.append("exclusive")
    _modular_py_test(tags = tags, **kwargs)

# NOTE: The OSS tree builds from source (SDLC-4549): the build macros no longer
# rewrite deps to the prebuilt `@modular_wheel`, no longer force the
# accelerator/profiling `defines` off (forcing MLRT_ACCELERATOR_SUPPORT=0 broke the
# from-source Mojo compiler build — KGENCompiler.cpp #include's the MLRT ErrorMacros
# header under `#if MLRT_ACCELERATOR_SUPPORT`), and no longer drop `internal_deps`
# (those are now in-repo source targets; the shared impls merge them into deps).
# Everything passes straight through to source, matching the internal build. The
# only remaining OSS-specific behavior is stubbing the not-yet-source-built profiler.
modular_cc_binary = _modular_cc_binary
modular_cc_test = _modular_cc_test

def modular_cc_library(name, **kwargs):
    """Builds a modular_cc_library from source, stubbing the profiler targets.

    Args:
        name: The target name.
        **kwargs: Passed through to modular_cc_library.
    """

    # TODO(SDLC-4549): the profiler (Tracy) is still stubbed in OSS; source-build
    # it in a follow-up (needs the tracy deps wired in OSS).
    if name == "Profiling":
        # Headers only, with every Tracy backend compiled out (no @tracy dep).
        _modular_cc_library(
            name = name,
            hdrs = kwargs.get("hdrs", []),
            defines = ["TRACY_CUDA_ENABLE=0", "TRACY_ROCM_ENABLE=0"],
            deps = ["//Support:TimeProfiler", "@llvm-project//llvm:Support"],
        )
        return
    if name == "ProfilerHostGlue":
        # Provide TimeProfiler for now since that may be what they're actually after
        _modular_cc_library(name = name, deps = ["//Support:TimeProfiler"])
        return

    _modular_cc_library(name = name, **kwargs)

def modular_shared_library(name = None, **kwargs):
    # TODO(SDLC-4549): profiler plugin still stubbed in OSS; see modular_cc_library.
    if name in ["MAXProfilerPlugin"]:
        # Provide TimeProfiler for now since that may be what they're actually after
        _modular_cc_library(name = name, deps = ["//Support:TimeProfiler"])
        return

    _modular_shared_library(name = name, **kwargs)

mojo_binary = _mojo_binary
mojo_library = _mojo_library
mojo_shared_library = _mojo_shared_library
modular_py_binary = _modular_py_binary
mef = _mef
copy_files = _copy_files

modular_nanobind_extension = _modular_nanobind_extension
modular_nanobind_library = _modular_nanobind_library

modular_generate_stubfiles = _modular_generate_stubfiles

def _noop(**_kwargs):
    pass

install_docs = _noop
modular_doxygen_docs = _noop
modular_python_binding_library_test = _noop

modular_pyo3_extension = _modular_pyo3_extension
