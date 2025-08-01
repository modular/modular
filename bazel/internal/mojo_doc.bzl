"""Generate documentation for Mojo source files."""

load("@rules_mojo//mojo:providers.bzl", "MojoInfo")
load("@rules_mojo//mojo/private:utils.bzl", "MOJO_EXTENSIONS", "collect_mojoinfo")  # buildifier: disable=bzl-visibility

def _mojo_doc_implementation(ctx):
    mojo_toolchain = ctx.toolchains["@rules_mojo//:toolchain_type"].mojo_toolchain_info

    import_paths, transitive_mojopkgs = collect_mojoinfo(ctx.attr.deps + mojo_toolchain.implicit_deps)
    root_directory = ctx.files.srcs[0].dirname

    file_args = ctx.actions.args()
    for file in ctx.files.srcs:
        if not file.dirname.startswith(root_directory):
            file_args.add("-I", file.dirname)

    file_args.add_all(import_paths, before_each = "-I")
    file_args.add(root_directory)

    mojodoc_output = ctx.actions.declare_file(ctx.label.name + ".mojodoc.json")
    ctx.actions.run(
        executable = mojo_toolchain.mojo,
        inputs = depset(ctx.files.srcs, transitive = [transitive_mojopkgs]),
        tools = mojo_toolchain.all_tools,
        outputs = [mojodoc_output],
        mnemonic = "MojoDoc",
        arguments = [
                        "doc",
                        "--validate-doc-strings",
                        "-o",
                        mojodoc_output.path,
                        file_args,
                    ] + (["--docs-base-path", ctx.attr.docs_base_path] if ctx.attr.docs_base_path else []) +
                    (["--diagnose-missing-doc-strings"] if ctx.attr.validate_missing_docs else []),
        progress_message = "%{label} generating mojodoc.json",
        env = {
            "MODULAR_HOME": ".",  # Make sure any cache files are written to somewhere bazel will cleanup
        },
        use_default_shell_env = True,
    )

    doc_output = ctx.actions.declare_directory(ctx.label.name)
    ctx.actions.run(
        executable = ctx.executable._mojodoc_json_to_markdown,
        outputs = [doc_output],
        inputs = [mojodoc_output],
        mnemonic = "MojoDoc",
        arguments = [
            "-o",
            doc_output.path,
            mojodoc_output.path,
        ],
        use_default_shell_env = True,
    )

    return [
        DefaultInfo(files = depset([doc_output])),
    ]

mojo_doc = rule(
    implementation = _mojo_doc_implementation,
    attrs = {
        "srcs": attr.label_list(
            allow_empty = False,
            allow_files = MOJO_EXTENSIONS,
        ),
        "deps": attr.label_list(
            providers = [MojoInfo],
        ),
        "validate_missing_docs": attr.bool(
            default = False,
            doc = "Fail for missing docstrings",
        ),
        "docs_base_path": attr.string(
            default = "",
            doc = "Base path prefix for generated documentation links",
        ),
        "_mojodoc_json_to_markdown": attr.label(
            default = Label("//bazel/internal:mojodoc_json_to_markdown"),
            cfg = "exec",
            executable = True,
        ),
    },
    toolchains = ["@rules_mojo//:toolchain_type"],
)
