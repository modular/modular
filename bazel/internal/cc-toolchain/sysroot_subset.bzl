"""Expose a subset of a sysroot's files for actions that don't need all of it."""

load("@bazel_skylib//rules/directory:providers.bzl", "DirectoryInfo")

def _sysroot_subset_implementation(ctx):
    directory = ctx.attr.sysroot[DirectoryInfo]

    # Off macOS the sysroot repository publishes an empty tree so that analysis
    # still succeeds on a platform that will never build against it. Globbing
    # that tree fails on the first pattern, so match the empty set instead.
    # Patterns are otherwise required to match, which is what catches a
    # pattern left stale by an SDK bump.
    if not directory.entries:
        return DefaultInfo(files = depset())

    return DefaultInfo(
        files = directory.glob(
            include = ctx.attr.include,
            exclude = ctx.attr.exclude,
        ),
    )

sysroot_subset = rule(
    implementation = _sysroot_subset_implementation,
    doc = """Files matching `include` within a sysroot, for use as `cc_args` data.

    A sysroot repository exposes its whole tree as one source directory, which
    is opaque: every action that lists it gets every file underneath, including
    the parts of the SDK nothing here compiles against. Locally that is one
    symlink, but remote execution has to put each file in the action's input
    tree, so the cost is paid per action.

    Globs run against the `directory` rule's tree rather than `native.glob`, so
    this works on a sysroot that lives in another repository.
    """,
    attrs = {
        "sysroot": attr.label(
            doc = "The bazel_skylib `directory` target for the sysroot root.",
            providers = [DirectoryInfo],
            mandatory = True,
        ),
        "include": attr.string_list(
            doc = "Globs to match, relative to the sysroot root.",
            mandatory = True,
        ),
        "exclude": attr.string_list(
            doc = "Globs to subtract from the matches of `include`.",
        ),
    },
)
