"""Create version status template file"""

def _status_template_impl(ctx):
    # FIXME(SDLC-528): Don't include the stable status here, *or* the unstable
    # status. The stable status unfortunately includes the BUILD_HOST.
    ctx.actions.run_shell(
        arguments = [
            ctx.files.src[0].path,
            ctx.outputs.out.path,
        ],
        command = 'cp -a "$1" "$2"',
        inputs = [
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
