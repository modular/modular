"""A rule to generate sphinx docs"""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")
load(":mojo_collect_deps_aspect.bzl", "collect_transitive_mojoinfo")
load(":mojo_test_environment.bzl", "mojo_test_environment")

def _sphinx_docs_impl(ctx):
    sources = set()
    outputs = []
    for src in ctx.attr.srcs:
        root = src.label.package
        for file in src.files.to_list():
            base_path = file.path.removeprefix(root + "/")
            output_name = base_path.removesuffix("rst") + "md"
            outputs.append(ctx.actions.declare_file("output/" + output_name))

            # This will be the input to `cp -r`, so we just need the first file/dir under the package prefix
            sources.add(root + "/" + base_path.split("/")[0])

    for src in ctx.attr.data:
        root = src.label.package
        for file in src.files.to_list():
            # This will be the input to `cp -r`, so we just need the first file/dir under the package prefix
            if file.path.startswith(root + "/"):
                base_path = file.path.removeprefix(root + "/")
                sources.add(root + "/" + base_path.split("/")[0])
            else:
                sources.add(file.dirname)

    command = """
    set -euo pipefail

    export MODULAR_HOME=$PWD

    # Gather inputs
    /bin/cp {config} ./conf.py
    /bin/mkdir sphinx_doc_inputs
    for s in {sources}; do
        /bin/cp -r $s sphinx_doc_inputs/
    done

    # Create outputs
    /bin/mkdir -p {output_dir}
    {sphinx} \\
        -q \\
        -b {builder} \\
        -c . \\
        -W \\
        ./sphinx_doc_inputs \\
        {output_dir}
    """.format(
        config = ctx.file.config_file.path,
        sphinx = ctx.executable._sphinx.path,
        builder = ctx.attr.builder,
        sources = " ".join(sorted(sources)),
        output_dir = ctx.bin_dir.path + "/" + ctx.label.package + "/output",
    )

    mojo_variables = ctx.attr.exec_test_environment[platform_common.TemplateVariableInfo].variables
    env = ctx.attr.env | {
        "MODULAR_MOJO_MAX_COMPILERRT_PATH": mojo_variables["COMPILER_RT_PATH"],
        "MODULAR_MOJO_MAX_DRIVER_PATH": mojo_variables["MOJO_BINARY_PATH"],
        "MODULAR_MOJO_MAX_IMPORT_PATH": mojo_variables["COMPUTED_IMPORT_PATH"],
        "MODULAR_MOJO_MAX_LINKER_DRIVER": mojo_variables["MOJO_LINKER_DRIVER"],
        "MODULAR_MOJO_MAX_SHARED_LIBS": mojo_variables["COMPUTED_LIBS"],
        "MODULAR_MOJO_MAX_SYSTEM_LIBS": mojo_variables["MOJO_LINKER_SYSTEM_LIBS"],
    }

    expanded_env = {"BAZEL_TEST": "1"}
    for key, value in env.items():
        expanded_env[key] = ctx.expand_make_variables(
            "env",
            ctx.expand_location(
                value,
                targets = ctx.attr.data + [ctx.attr._sphinx],
            ),
            {},
        )

    ctx.actions.run_shell(
        inputs = [ctx.file.config_file] + ctx.files.srcs + ctx.files.data + ctx.attr._sphinx[DefaultInfo].default_runfiles.files.to_list(),
        outputs = outputs,
        command = command,
        tools = [ctx.executable._sphinx],
        env = expanded_env,
        use_default_shell_env = True,
    )

    return [
        DefaultInfo(
            files = depset(outputs),
        ),
    ]

_modular_sphinx_docs = rule(
    implementation = _sphinx_docs_impl,
    attrs = {
        "builder": attr.string(values = ["html", "markdown"], mandatory = True),
        "config_file": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "conf.py target",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "data": attr.label_list(
            allow_files = True,
        ),
        "env": attr.string_dict(),
        "exec_test_environment": attr.label(
            cfg = CFG_WORKAROUND,
        ),
        "_sphinx": attr.label(
            default = Label("//bazel/internal:sphinx"),
            executable = True,
            cfg = CFG_WORKAROUND,
        ),
    },
    toolchains = [
        # Force to run on target platform
        "@bazel_tools//tools/test:default_test_toolchain_type",
    ],
)

def modular_sphinx_docs(name, **kwargs):
    """
    A rule to generate sphinx docs.

    Args:
        name: The name of the rule.
        **kwargs: Additional arguments to pass to the rule.
    """

    transitive_mojo_deps = name + ".mojo_deps"
    collect_transitive_mojoinfo(
        name = transitive_mojo_deps,
        deps_to_scan = ["@//bazel/internal:sphinx"],
    )

    env_name = name + ".mojo_test_env"
    mojo_test_environment(
        name = env_name,
        data = [transitive_mojo_deps],
        short_path = False,
    )

    _modular_sphinx_docs(
        name = name,
        exec_test_environment = env_name,
        **kwargs
    )
