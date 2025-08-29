"""A rule to run a tool with the necessary environment variables.

This is useful for running tools directly, with their implicit dependencies set
in the environment, in order to test things outside of bazel
"""

load("@rules_python//python:py_info.bzl", "PyInfo")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "env_for_available_tools")  # buildifier: disable=bzl-visibility
load(":mojo_collect_deps_aspect.bzl", "collect_transitive_mojoinfo")  # buildifier: disable=bzl-visibility
load(":mojo_test_environment.bzl", "mojo_test_environment")  # buildifier: disable=bzl-visibility

def _py_repl_impl(ctx):
    toolchain = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"]
    python_runfiles = [
        ctx.runfiles(files = toolchain.py3_runtime.files.to_list()),
    ]

    python_imports = []
    for dep in ctx.attr.deps:
        python_imports.append(dep[PyInfo].imports)
        python_runfiles.extend([
            ctx.runfiles(files = dep[PyInfo].transitive_sources.to_list()),
            dep[DefaultInfo].default_runfiles,
        ])

    python_path = ""
    for path in depset(transitive = python_imports).to_list():
        python_path += "../" + path + ":"

    # https://github.com/bazelbuild/rules_python/issues/2262
    libpython = None
    for file in toolchain.py3_runtime.files.to_list():
        if file.basename.startswith("libpython"):
            libpython = file.short_path

            # if there are multiple any of them should work and they are likely symlinks to each other
            break

    if not libpython:
        fail("Failed to find libpython")

    expanded_env = {
        "PYTHON3": str(toolchain.py3_runtime.interpreter.short_path),
        "MOJO_PYTHON_LIBRARY": libpython,
    }
    if python_path:
        expanded_env["PYTHONPATH"] = python_path

    is_linux = ctx.target_platform_has_constraint(ctx.attr._linux_constraint[platform_common.ConstraintValueInfo])
    env = ctx.attr.env | env_for_available_tools(os = "linux" if is_linux else "macos", location_specifier = "execpath")
    for key, value in env.items():
        expanded_env[key] = ctx.expand_make_variables(
            "env",
            ctx.expand_location(
                value,
                targets = ctx.attr.deps + ctx.attr.data,
            ),
            {},
        )

    output = ctx.actions.declare_file(ctx.attr.name)
    ctx.actions.symlink(
        output = output,
        target_file = toolchain.py3_runtime.interpreter,
        is_executable = True,
    )

    return [
        DefaultInfo(
            executable = output,
            files = depset([output]),
            runfiles = ctx.runfiles(ctx.files.data + ctx.files.srcs).merge_all(python_runfiles),
        ),
        RunEnvironmentInfo(
            environment = expanded_env,
            inherited_environment = ctx.attr.env_inherit,
        ),
    ]

_py_repl = rule(
    implementation = _py_repl_impl,
    attrs = {
        "data": attr.label_list(allow_files = True),
        "srcs": attr.label_list(allow_files = [".py"]),
        "deps": attr.label_list(providers = [PyInfo]),
        "env": attr.string_dict(
            doc = "Environment variables set before running the tool",
        ),
        "env_inherit": attr.string_list(
            doc = "Environment variables to inherit from the global env",
        ),
        "_linux_constraint": attr.label(
            default = Label("@platforms//os:linux"),
        ),
    },
    toolchains = [
        "@bazel_tools//tools/python:toolchain_type",
    ],
    executable = True,
    doc = """
Create a bazel runnable target that launches a python repl with the given deps
""",
)

def py_repl(
        name,
        data = [],
        deps = [],
        env = {},
        toolchains = [],
        direct = True,
        **kwargs):
    """Create a target that drops you into a python repl with the given deps.

    Args:
        name: The name of the target
        data: Runtime deps of the target
        deps: Python deps of the target
        env: Any environment variables that should be set in the repl
        toolchains: See upstream py_binary docs
        direct: True if this is called from a BUILD file
        **kwargs: Extra arguments passed through to py_binary
    """
    extra_toolchains = []
    extra_data = []
    extra_env = {}

    if "//bazel/internal:lib_toolchain" not in toolchains and "@//bazel/internal:lib_toolchain" not in toolchains:
        extra_toolchains.append("@//bazel/internal:lib_toolchain")

    if direct:
        transitive_mojo_deps = name + ".mojo_deps"
        collect_transitive_mojoinfo(
            name = transitive_mojo_deps,
            deps_to_scan = deps,
            testonly = True,
            tags = [ALLOW_UNUSED_TAG],
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
            data = [transitive_mojo_deps],
            short_path = True,
            testonly = True,
            tags = [ALLOW_UNUSED_TAG],
        )

    _py_repl(
        name = name,
        data = extra_data + data,
        deps = deps + [
            "@//bazel/internal:bazel_sitecustomize",
        ],
        toolchains = toolchains + extra_toolchains,
        env = extra_env | env,
        testonly = True,
        tags = ["manual", ALLOW_UNUSED_TAG],
        **kwargs
    )
