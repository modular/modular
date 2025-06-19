"""A rule to run a given binary as a test."""

def _binary_test_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(
        target_file = ctx.executable.binary,
        output = output,
    )

    processed_env = {}
    for key, value in ctx.attr.env.items():
        processed_env[key] = ctx.expand_make_variables(
            "env",
            ctx.expand_location(value, targets = ctx.attr.data),
            {},
        )

    env_inherit = ctx.attr.env_inherit + ctx.attr.binary[RunEnvironmentInfo].inherited_environment

    extra_providers = []
    transitive_files = depset()
    if InstrumentedFilesInfo in ctx.attr.binary:
        provider = ctx.attr.binary[InstrumentedFilesInfo]
        transitive_files = provider.instrumented_files
        extra_providers.append(provider)

    return [
        DefaultInfo(
            executable = output,
            runfiles = ctx.runfiles(ctx.files.data, transitive_files = transitive_files)
                .merge(ctx.attr.binary[DefaultInfo].default_runfiles),
        ),
        RunEnvironmentInfo(
            environment = ctx.attr.binary[RunEnvironmentInfo].environment | processed_env,
            inherited_environment = env_inherit,
        ),
    ] + extra_providers

binary_test = rule(
    implementation = _binary_test_impl,
    attrs = {
        "binary": attr.label(
            executable = True,
            cfg = "target",
        ),
        "env": attr.string_dict(),
        "data": attr.label_list(allow_files = True),
        "env_inherit": attr.string_list(),
    },
    test = True,
)
