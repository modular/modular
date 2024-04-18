"""Create version status template file"""

def _status_template_impl(ctx):
    ctx.actions.run_shell(
        arguments = [
            ctx.info_file.path,
            ctx.version_file.path,
            ctx.files.src[0].path,
            ctx.outputs.out.path,
        ],
        command = 'cat "$3" > "$4" && (cat "$1" "$2" | while read var value; do sed -e "s|\\${${var}}|${value}|g" -i.bak "$4"; done)',
        inputs = [
            ctx.info_file,
            ctx.version_file,
            ctx.files.src[0],
        ],
        outputs = [ctx.outputs.out],
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

status_template = rule(
    implementation = _status_template_impl,
    output_to_genfiles = True,
    attrs = {
        "out": attr.output(doc = "Output file for the rendered input."),
        "src": attr.label(doc = "Input template (using simple shell expansion).", allow_single_file = True),
    },
)
