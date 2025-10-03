"""A toolchain to return the library extension for make variable substitution."""

def _impl(ctx):
    is_linux = ctx.target_platform_has_constraint(ctx.attr._linux_constraint[platform_common.ConstraintValueInfo])
    if is_linux:
        lib_extension = "so"
    else:
        lib_extension = "dylib"

    return [
        platform_common.TemplateVariableInfo({
            "LIB_EXTENSION": lib_extension,
        }),
    ]

lib_toolchain = rule(
    implementation = _impl,
    attrs = {
        "_linux_constraint": attr.label(
            default = Label("@platforms//os:linux"),
        ),
    },
)
