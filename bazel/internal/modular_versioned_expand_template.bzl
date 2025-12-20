"""expand_template with additional substitutions for Modular release versions."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _modular_versioned_expand_template_impl(ctx):
    major = ctx.attr._modular_version_major[BuildSettingInfo].value
    minor = ctx.attr._modular_version_minor[BuildSettingInfo].value
    patch = ctx.attr._modular_version_patch[BuildSettingInfo].value
    label = ctx.attr._modular_version_label[BuildSettingInfo].value
    release_type = ctx.attr._release_type[BuildSettingInfo].value
    sha = ctx.attr._modular_version_sha[BuildSettingInfo].value

    expanded_substitutions = {}
    for key, value in ctx.attr.substitutions.items():
        expanded_substitutions[key] = ctx.expand_make_variables(
            "substitutions",
            ctx.expand_location(
                value,
                targets = ctx.attr.data,
            ),
            {},
        )

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = expanded_substitutions | {
            "@MODULAR_VERSION_MAJOR@": major,
            "@MODULAR_VERSION_MINOR@": minor,
            "@MODULAR_VERSION_PATCH@": patch,
            "@MODULAR_VERSION_REVISION@": sha,
            "@MODULAR_VERSION_LABEL@": label,
            "@MODULAR_BUILD_TYPE_LOWER@": release_type,
            "@MODULAR_VERSION_STRING@": "{}.{}.{}{}".format(
                major,
                minor,
                patch,
                label,
            ),
        },
    )

modular_versioned_expand_template = rule(
    implementation = _modular_versioned_expand_template_impl,
    doc = """Template expansion

This performs a simple search over the template file for the keys in
substitutions, and replaces them with the corresponding values.

There is no special syntax for the keys. To avoid conflicts, you would need to
explicitly add delimiters to the key strings, for example "{KEY}" or "@KEY@".""",
    attrs = {
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "The template file to expand.",
        ),
        "substitutions": attr.string_dict(
            mandatory = True,
            doc = "A dictionary mapping strings to their substitutions.",
        ),
        "data": attr.label_list(
            allow_files = True,
            doc = "data dependencies. See" +
                  " https://bazel.build/reference/be/common-definitions#typical.data",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "The destination of the expanded file.",
        ),
        "_modular_version_major": attr.label(default = "//:modular_version_major"),
        "_modular_version_minor": attr.label(default = "//:modular_version_minor"),
        "_modular_version_patch": attr.label(default = "//:modular_version_patch"),
        "_modular_version_label": attr.label(default = "//:modular_version_label"),
        "_modular_version_sha": attr.label(default = "//:modular_version_sha"),
        "_release_type": attr.label(default = "//:release_type"),
    },
)
