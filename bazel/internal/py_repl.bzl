"""A rule to run a tool with the necessary environment variables.

This is useful for running tools directly, with their implicit dependencies set
in the environment, in order to test things outside of bazel
"""

load("@rules_python//python:py_info.bzl", "PyInfo")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "env_for_available_tools")  # buildifier: disable=bzl-visibility

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
        deps = [],
        tags = [],
        toolchains = [],
        **kwargs):
    extra_toolchains = []
    if "//bazel/internal:lib_toolchain" not in toolchains and "@//bazel/internal:lib_toolchain" not in toolchains:
        extra_toolchains.append("@//bazel/internal:lib_toolchain")

    _py_repl(
        name = name,
        deps = deps + [
            "@//bazel/internal:bazel_sitecustomize",
        ],
        toolchains = toolchains + extra_toolchains,
        tags = tags + [ALLOW_UNUSED_TAG],
        **kwargs
    )
