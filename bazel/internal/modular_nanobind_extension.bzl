"""Wrapper for the upstream nanobind_extension rule to apply modular specific defaults."""

load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load(":modular_shared_library.bzl", "modular_shared_library")

def _python_extension_impl(ctx):
    # Beside the library, so its $ORIGIN-relative rpaths still resolve.
    file_name = ctx.expand_make_variables("file_name", ctx.attr.file_name, {})
    if ctx.attr.directory:
        file_name = ctx.attr.directory + "/" + file_name
    extension = ctx.actions.declare_file(file_name)
    ctx.actions.symlink(output = extension, target_file = ctx.file.shared_library)
    return [DefaultInfo(
        files = depset([extension]),
        runfiles = ctx.runfiles(files = [extension]).merge_all([
            ctx.attr.shared_library[DefaultInfo].default_runfiles,
            ctx.attr.static[DefaultInfo].default_runfiles,
        ]),
    )]

# Gives the shared library the file name Python imports, which
# cc_shared_library can't compute: it doesn't expand make variables.
_python_extension = rule(
    implementation = _python_extension_impl,
    attrs = {
        "directory": attr.string(),
        "file_name": attr.string(mandatory = True),
        "shared_library": attr.label(mandatory = True, allow_single_file = True),
        "static": attr.label(mandatory = True),
    },
)

def modular_nanobind_extension(
        name,
        new_link_root = None,
        copts = [],
        defines = [],
        deps = [],
        linkopts = [],
        module_name = None,
        tags = [],
        testonly = False,
        visibility = None,
        **kwargs):
    """A Python extension module built with nanobind

    `name` is the module file, `name.stripped` the same file stripped, and
    `name.dso` the modular_shared_library behind them, whose `dynamic_deps` it
    takes.

    Args:
        name: The name of the core target, the prefix of the name of the other python specific targets
        new_link_root: If specified, relative path from the module in the package to the package's top level `lib` folder.
        copts: See cc_binary docs
        defines: See cc_binary docs
        deps: See cc_binary docs
        linkopts: See cc_binary docs
        module_name: Specifies the name of the binding module. Defaults to `name` if None.
        tags: Set on all underlying targets
        testonly: Set on all underlying targets
        visibility: Set on all underlying targets
        **kwargs: Everything else passed through to modular_shared_library
    """
    link_root_arg = select({
        "//conditions:default": [],
        "@platforms//os:linux": ["-Wl,-rpath,$ORIGIN/" + new_link_root],
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path/" + new_link_root],
    }) if new_link_root else []

    modular_shared_library(
        name = name + ".dso",
        copts = copts + [
            "-fexceptions",
            "-fno-stack-protector",
            "-fno-strict-aliasing",
            "-frtti",
        ],
        linkopts = linkopts + select({
            # libpython symbols should be available at runtime
            # NOTE: Force gc-sections/dead_strip to avoid nanobind linking issue
            "@platforms//os:linux": [
                "-Wl,-z,undefs",
                "-Wl,--gc-sections",
            ],
            "@platforms//os:macos": [
                "-Wl,-undefined,dynamic_lookup",
                "-Wl,-dead_strip",
            ],
            "//conditions:default": [],
        }) + link_root_arg,
        defines = defines + select({
            "@rules_python//python/config_settings:is_py_freethreaded": ["NB_FREE_THREADED"],
            "//conditions:default": [],
        }),
        target_compatible_with = select({
            # sanitizer libraries aren't available when this library is loaded by python
            "//:asan_macos": ["@platforms//:incompatible"],
            "//:tsan": ["@platforms//:incompatible"],
            "//:ubsan": ["@platforms//:incompatible"],
            "//conditions:default": [],
        }),
        deps = deps + [
            "@rules_python//python/cc:current_py_cc_headers",
            "@nanobind",
        ],
        tags = tags,
        testonly = testonly,
        visibility = visibility,
        **kwargs
    )

    _python_extension(
        name = name,
        file_name = "{}.$(PYTHON_EXTENSION_NAME)".format(module_name or name),
        shared_library = ":" + name + ".dso",
        static = ":" + name + ".dso.static",
        toolchains = ["//bazel/internal:python_extension_name_toolchain"],
        tags = tags,
        testonly = testonly,
        visibility = visibility,
    )

    _python_extension(
        name = name + ".stripped",
        directory = name + ".stripped",
        file_name = "{}.$(PYTHON_EXTENSION_NAME)".format(module_name or name),
        shared_library = ":" + name + ".dso.stripped",
        static = ":" + name + ".dso.static",
        toolchains = ["//bazel/internal:python_extension_name_toolchain"],
        # Only release packaging reads it, and not every extension ships.
        tags = tags + ["manual", ALLOW_UNUSED_TAG],
        testonly = testonly,
        visibility = visibility,
    )
