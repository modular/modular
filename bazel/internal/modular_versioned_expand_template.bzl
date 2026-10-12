"""expand_template with additional substitutions for Modular release versions."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _modular_versioned_expand_template_impl(ctx):
    max_base_version = ctx.attr._max_base_version[BuildSettingInfo].value
    max_major, max_minor, max_patch = max_base_version.split(".")
    max_label = ctx.attr._max_version_label[BuildSettingInfo].value

    mojo_base_version = ctx.attr._mojo_base_version[BuildSettingInfo].value

    mojo_major, mojo_minor, mojo_patch = mojo_base_version.split(".")
    mojo_label = ctx.attr._mojo_version_label[BuildSettingInfo].value

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
            "@MAX_VERSION_MAJOR@": max_major,
            "@MAX_VERSION_MINOR@": max_minor,
            "@MAX_VERSION_PATCH@": max_patch,
            "@MAX_VERSION_LABEL@": max_label,
            "@MAX_VERSION_STRING@": max_base_version + max_label,
            "@MODULAR_VERSION_REVISION@": sha,
            "@MODULAR_BUILD_TYPE_LOWER@": release_type,
            "@MOJO_VERSION_MAJOR@": mojo_major,
            "@MOJO_VERSION_MINOR@": mojo_minor,
            "@MOJO_VERSION_PATCH@": mojo_patch,
            "@MOJO_VERSION_LABEL@": mojo_label,
            "@MOJO_VERSION_STRING@": mojo_base_version + mojo_label,

            # For backward compatibility, TODO REMOVE
            "@MODULAR_VERSION_MAJOR@": max_major,
            "@MODULAR_VERSION_MINOR@": max_minor,
            "@MODULAR_VERSION_PATCH@": max_patch,
            "@MODULAR_VERSION_LABEL@": max_label,
            "@MODULAR_VERSION_STRING@": max_base_version + max_label,
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
        "_max_base_version": attr.label(default = "//:max_base_version"),
        "_mojo_base_version": attr.label(default = "//:mojo_base_version"),
        "_max_version_label": attr.label(default = "//:max_version_label"),
        "_mojo_version_label": attr.label(default = "//:mojo_version_label"),
        "_modular_version_sha": attr.label(default = "//:modular_version_sha"),
        "_release_type": attr.label(default = "//:release_type"),
    },
)
