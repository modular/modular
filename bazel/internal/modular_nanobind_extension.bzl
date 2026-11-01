"""Wrapper for the upstream nanobind_extension rule to apply modular specific defaults."""

load(":modular_shared_library.bzl", "modular_shared_library")

def modular_nanobind_extension(
        name,
        new_link_root = None,
        copts = [],
        defines = [],
        deps = [],
        linkopts = [],
        module_name = None,
        **kwargs):
    """A wrapper for cc_binary

    https://bazel.build/reference/be/c-cpp#cc_binary

    Args:
        name: The name of the core target, the prefix of the name of the other python specific targets
        new_link_root: If specified, relative path from the module in the package to the package's top level `lib` folder.
        copts: See cc_binary docs
        defines: See cc_binary docs
        deps: See cc_binary docs
        linkopts: See cc_binary docs
        module_name: Specifies the name of the binding module. Defaults to `name` if None.
        **kwargs: Everything else passed through to cc_binary without processing
    """
    link_root_arg = select({
        "//conditions:default": [],
        "@platforms//os:linux": ["-Wl,-rpath,$ORIGIN/" + new_link_root],
        "@platforms//os:macos": ["-Wl,-rpath,@loader_path/" + new_link_root],
    }) if new_link_root else []

    modular_shared_library(
        name = name,
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
        shared_lib_name = "{}.$(PYTHON_EXTENSION_NAME)".format(module_name or name),
        toolchains = ["//bazel/internal:python_extension_name_toolchain"],
        **kwargs
    )
