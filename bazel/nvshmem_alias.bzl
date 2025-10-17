"""Alias for nvshmem."""

def _nvshmem_alias_impl(rctx):
    rctx.file("BUILD.bazel", content = """\
package(default_visibility = ["//visibility:public"])

alias(
    name = "hostlibs",
    actual = "{target}:hostlibs",
)
""".format(target = rctx.attr.target))

nvshmem_alias = repository_rule(
    implementation = _nvshmem_alias_impl,
    attrs = {
        "target": attr.string(
            doc = "The real target of nvshmem",
            default = "@@//open-source/max/bazel/third-party/nvshmem",
        ),
    },
)
