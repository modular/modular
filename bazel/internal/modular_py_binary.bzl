"""A helper macro for python scripts which helps setup various runtime dependencies."""

load("@rules_python//python:defs.bzl", "py_binary")
load("//bazel/internal:config.bzl", "env_for_available_tools")  # buildifier: disable=bzl-visibility
load(":modular_py_venv.bzl", "modular_py_venv")
load(":mojo_collect_deps_aspect.bzl", "collect_transitive_mojoinfo")
load(":mojo_test_environment.bzl", "mojo_test_environment")
load(":py_repl.bzl", "py_repl")

def modular_py_binary(
        name,
        srcs,
        main = None,
        env = {},
        data = [],
        deps = [],
        mojo_deps = [],
        toolchains = [],
        imports = [],
        tags = [],
        args = [],
        testonly = False,
        **kwargs):
    """Creates a pytest based python test target.

    Args:
        name: The name of the test target
        srcs: The test source files
        main: See upstream py_binary docs
        env: Any environment variables that should be set during the test runtime
        data: Runtime deps of the test target
        deps: Python deps of the target
        mojo_deps: mojo_library targets the test depends on at runtime
        toolchains: See upstream py_binary docs
        imports: See upstream py_binary docs
        tags: See upstream py_binary docs
        args: See upstream py_binary docs
        testonly: Only test targets can depend on this target
        **kwargs: Extra arguments passed through to py_binary
    """
    extra_toolchains = [
        "@//bazel/internal:lib_toolchain",
    ]
    extra_env = {}
    extra_data = []
    extra_deps = []

    transitive_mojo_deps = name + ".mojo_deps"
    collect_transitive_mojoinfo(
        name = transitive_mojo_deps,
        deps_to_scan = deps,
        testonly = testonly,
    )

    env_name = name + ".mojo_test_env"
    extra_toolchains.append(env_name)
    extra_data.append(env_name)
    extra_env |= {
        "MODULAR_MOJO_MAX_COMPILERRT_PATH": "$(COMPILER_RT_PATH)",
        "MODULAR_MOJO_MAX_DRIVER_PATH": "$(MOJO_BINARY_PATH)",
        "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
        "MODULAR_MOJO_MAX_LINKER_DRIVER": "$(MOJO_LINKER_DRIVER)",
        "MODULAR_MOJO_MAX_LLD_PATH": "$(LLD_PATH)",
        "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
    }
    mojo_test_environment(
        name = env_name,
        data = mojo_deps + [transitive_mojo_deps],
        short_path = True,
        testonly = testonly,
    )

    py_binary(
        name = name,
        data = extra_data + data,
        deps = deps + extra_deps + [
            "@//bazel/internal:bazel_sitecustomize",  # py_repl adds this automatically
        ],
        srcs = srcs,
        main = main,
        env = env_for_available_tools() | extra_env | env | {"MODULAR_CANNOT_DEBUG": "1"},
        toolchains = extra_toolchains + toolchains,
        imports = imports,
        tags = tags,
        args = args,
        testonly = testonly,
        **kwargs
    )

    modular_py_venv(
        name = name + ".venv",
        data = extra_data + data,
        deps = deps + extra_deps + [
            "@//bazel/internal:bazel_sitecustomize",  # py_repl adds this automatically
        ],
    )

    py_repl(
        name = name + ".debug",
        data = extra_data + data,
        deps = deps + extra_deps,
        direct = False,
        env = env_for_available_tools(location_specifier = "execpath") | extra_env | env,
        args = [native.package_name() + "/" + (main or srcs[0])],
        srcs = srcs,
        toolchains = extra_toolchains + toolchains,
        **kwargs
    )
