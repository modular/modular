"""Alias for mblack."""

def _mblack_alias_impl(rctx):
    rctx.file("BUILD.bazel", content = """\
package(default_visibility = ["//visibility:public"])

alias(
    name = "mblack",
    actual = "{target}",
)
""".format(target = rctx.attr.target))

mblack_alias = repository_rule(
    implementation = _mblack_alias_impl,
    attrs = {
        "target": attr.string(
            doc = "The real target of mblack",
            default = "@modular_wheel//:mblack",
        ),
    },
)
