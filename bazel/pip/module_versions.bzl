"""A repo rule to expose constants that have to be hardcoded in the MODULE.bazel to other bzl files."""

def _impl(rctx):
    rctx.file("BUILD.bazel", "")
    rctx.file("config.bzl", """\
DEFAULT_PYTHON_VERSION = "{default_python_version}"
DEFAULT_PYTHON_VERSION_UNDERBAR = "{default_python_version}".replace(".", "_")
DEFAULT_PYTHON_VERSION_NOSEP = "{default_python_version}".replace(".", "")

PYTHON_VERSIONS_UNDERBAR = {python_versions}
PYTHON_VERSIONS_DOTTED = [x.replace("_", ".") for x in PYTHON_VERSIONS_UNDERBAR]
""".format(
        default_python_version = rctx.attr.default_python_version,
        python_versions = str(rctx.attr.python_versions),
    ))

module_versions = repository_rule(
    implementation = _impl,
    attrs = {
        "default_python_version": attr.string(mandatory = True),
        "python_versions": attr.string_list(mandatory = True),
    },
)
