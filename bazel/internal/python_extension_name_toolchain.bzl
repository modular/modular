"""Format the suffix of a native Python extension module name based on the Python version and platform."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

def _impl(ctx):
    py_version = ctx.attr._py_version[BuildSettingInfo].value
    is_freethreaded = ctx.attr._is_freethreaded[BuildSettingInfo].value == "yes"  # NOTE: Value is "yes" or "no", enforced
    suffix = "t" if is_freethreaded else ""
    return [
        platform_common.TemplateVariableInfo({
            "PYTHON_EXTENSION_NAME": "cpython-{}{}-{}.so".format(
                py_version.replace(".", ""),
                suffix,
                ctx.attr.platform,
            ),
        }),
    ]

python_extension_name_toolchain = rule(
    implementation = _impl,
    attrs = {
        "platform": attr.string(),
        "_py_version": attr.label(default = Label("@rules_python//python/config_settings:python_version")),
        "_is_freethreaded": attr.label(default = Label("@rules_python//python/config_settings:py_freethreaded")),
    },
)
