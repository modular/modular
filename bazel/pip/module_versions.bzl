"""A repo rule to expose constants that have to be hardcoded in the MODULE.bazel to other bzl files."""

def _impl(rctx):
    rctx.file("BUILD.bazel", "")
    rctx.file("config.bzl", """\
DEFAULT_PYTHON_VERSION = "{default_python_version}"
DEFAULT_PYTHON_VERSION_UNDERBAR = "{default_python_version}".replace(".", "_")
DEFAULT_PYTHON_VERSION_NOSEP = "{default_python_version}".replace(".", "")
PYTHON_VERSIONS = {python_versions}

TORCH_DEFAULT_VERSION = "{default_torch_version}"
TORCH_VERSIONS = {torch_versions}
""".format(
        default_python_version = rctx.attr.default_python_version,
        default_torch_version = rctx.attr.default_torch_version,
        python_versions = str(rctx.attr.python_versions),
        torch_versions = str(rctx.attr.torch_versions),
    ))

module_versions = repository_rule(
    implementation = _impl,
    attrs = {
        "default_python_version": attr.string(mandatory = True),
        "default_torch_version": attr.string(mandatory = True),
        "python_versions": attr.string_list(mandatory = True),
        "torch_versions": attr.string_list(mandatory = True),
    },
)
