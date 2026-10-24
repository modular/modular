"""Generate tblgen source from the tblgen json dumps."""

def _tblgen_nanobind_impl(ctx):
    defined_in = ["--defined-in", ctx.attr.defined_in] if ctx.attr.defined_in else []
    dialect = ["--dialect", ctx.attr.dialect] if ctx.attr.dialect else []

    inputs = [ctx.file.src]
    allowlist = []
    if ctx.file.allowlist:
        inputs.append(ctx.file.allowlist)
        allowlist = ["--allowlist", ctx.file.allowlist.path]

    output = ctx.outputs.output
    if not output:
        output = ctx.actions.declare_file(ctx.label.name + ".cpp.inc")

    ctx.actions.run(
        tools = [ctx.executable.tblgen],
        executable = ctx.executable.tblgen,
        inputs = inputs,
        arguments = [ctx.file.src.path] + defined_in + dialect + allowlist + ctx.attr.names + ["-o", output.path],
        outputs = [output],
    )

    return [DefaultInfo(files = depset([output]))]

tblgen_nanobind = rule(
    _tblgen_nanobind_impl,
    doc = "Generates tabular code from a tblgen json dump.",
    attrs = {
        "tblgen": attr.label(
            cfg = "exec",
            executable = True,
            default = "//max/python/max/_core/max_license/mlir_nanobind/tblgen",
        ),
        "src": attr.label(allow_single_file = [".json"]),
        "output": attr.output(),
        "defined_in": attr.string(),
        "dialect": attr.string(),
        "names": attr.string_list(),
        "allowlist": attr.label(allow_single_file = True),
    },
)

def mlir_nanobind(name, srcs, deps, output_dir, gentbl_filegroup, dialect = None, allowlist = None, **kwargs):
    """Generate nanobind bindings for multiple dialect files.

    Args:
        name: The target name for the generated files.
        srcs: .td files to use for generation.
        deps: .td dependencies.
        output_dir: The path where generated files will go. These will
            be used to specify the include paths.
        gentbl_filegroup: The TableGen filegroup rule to use. Passed in by
            //bazel:tablegen.bzl, whose path differs between the internal
            and open-source trees, so this file can't load it itself.
        dialect: If specified, limit generation to these specific dialects.
        allowlist: A file naming the symbols to generate, one per line. It has
            to name everything those bindings refer to as well; generation
            fails otherwise. Generates everything when unset.
        **kwargs: Forwarded to the finale filegroup definition.
    """
    outputs = []
    for src in srcs:
        basename = src.rsplit("/", 1)[1].rsplit(".")[0]
        json_target = basename + ".json"
        json_file = output_dir + "/" + json_target
        cinc_target = basename + ".cpp.inc"
        cinc_file = output_dir + "/" + cinc_target

        gentbl_filegroup(
            name = json_target,
            tbl_outs = [(["--dump-json"], json_file)],
            tblgen = "@llvm-project//llvm:llvm-tblgen",
            td_file = src,
            deps = deps,
        )

        tblgen_nanobind(
            name = cinc_target,
            src = json_target,
            defined_in = basename + ".td",
            dialect = dialect,
            allowlist = allowlist,
            output = cinc_file,
        )

        outputs.append(cinc_file)

    native.filegroup(
        name = name,
        srcs = outputs,
        **kwargs
    )
