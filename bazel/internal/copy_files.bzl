"""A rule to copy multiple files inferring their destination based on the input basenames"""

def _copy_files_impl(ctx):
    for src in ctx.files.srcs:
        potential_outputs = []
        for out in ctx.outputs.outs:
            if out.basename == src.basename:
                potential_outputs.append(out)

        if len(potential_outputs) != 1:
            fail("invalid outputs {} for input {}".format(potential_outputs, src))

        output = potential_outputs[0]

        ctx.actions.run_shell(
            inputs = [src],
            outputs = [output],
            command = "cp -f \"$1\" \"$2\"",
            arguments = [src.path, output.path],
            mnemonic = "CopyFile",
            progress_message = "Copying files",
            use_default_shell_env = True,
        )

copy_files = rule(
    implementation = _copy_files_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "outs": attr.output_list(),
    },
)
