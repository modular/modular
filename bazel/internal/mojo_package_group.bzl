"""`mojo_package_group`: one label standing for a set of Mojo packages."""

load("@rules_mojo//mojo:providers.bzl", "MojoInfo")
load("@rules_mojo//mojo/private:utils.bzl", "collect_mojoinfo")  # buildifier: disable=bzl-visibility

def _mojo_package_group_implementation(ctx):
    import_paths, mojodeps, ccdeps = collect_mojoinfo(ctx.attr.deps)
    return [
        DefaultInfo(
            # The group's own packages, not the whole transitive closure, so a
            # `data` or `filegroup` consumer gets what it named.
            files = depset(transitive = [
                dep[DefaultInfo].files
                for dep in ctx.attr.deps
            ]),
            runfiles = ctx.runfiles().merge_all([
                dep[DefaultInfo].default_runfiles
                for dep in ctx.attr.deps
            ]),
        ),
        MojoInfo(
            import_paths = import_paths,
            mojodeps = mojodeps,
            ccdeps = ccdeps,
        ),
    ]

mojo_package_group = rule(
    implementation = _mojo_package_group_implementation,
    doc = """Groups Mojo packages behind one label.

Depending on the group is the same as depending on every package in it: it
forwards their `MojoInfo` unchanged and compiles nothing of its own. A package
split into several, as `builtin_kernels` is, keeps one name for the targets
that want all of it.

It is not a Mojo package and has no import name. A target that needs one file
per package -- anything taking `$(location)` of each, or installing them --
names the packages instead.
""",
    attrs = {
        "deps": attr.label_list(
            providers = [MojoInfo],
            doc = "The packages this label stands for.",
        ),
    },
)
