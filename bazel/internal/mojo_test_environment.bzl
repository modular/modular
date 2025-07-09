"""Fetch the environment variables that need to be set to execute Mojo during a test."""

load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_mojo//mojo:providers.bzl", "MojoInfo")
load("@rules_mojo//mojo/private:utils.bzl", "collect_mojoinfo")  # buildifier: disable=bzl-visibility
load("@rules_python//python:py_info.bzl", "PyInfo")

def _mojo_test_environment_implementation(ctx):
    _, transitive_mojopkgs = collect_mojoinfo(ctx.attr.data)
    if not transitive_mojopkgs:
        return [
            CcInfo(),  # Requirement of py_test
            PyInfo(transitive_sources = depset()),  # Requirement of py_test
            platform_common.TemplateVariableInfo({
                "COMPUTED_IMPORT_PATH": "",
                "COMPUTED_LIBS": "",
            }),
        ]

    # The import_paths when used as runfiles like this differs from the standard ones
    import_paths = sets.make()
    for pkg in transitive_mojopkgs.to_list():
        if ctx.attr.short_path:
            sets.insert(import_paths, paths.dirname(pkg.short_path))
        else:
            sets.insert(import_paths, paths.dirname(pkg.path))

    transitive_runfiles = []
    for target in ctx.attr.data:
        transitive_runfiles.append(target[DefaultInfo].default_runfiles)

    shared_libs = []
    transitive_libs = []
    for lib in ctx.attr.libs:
        file = lib[DefaultInfo].files.to_list()[0]
        transitive_libs.append(file)
        path = file.path
        if ctx.attr.short_path:
            path = file.short_path

        shared_libs.append(path)
        shared_libs.append("-Xlinker,-rpath,-Xlinker,{}".format(paths.dirname(path)))

    return [
        CcInfo(),  # Requirement of py_test
        PyInfo(transitive_sources = depset()),  # Requirement of py_test
        DefaultInfo(
            runfiles = ctx.runfiles(
                transitive_files = depset(transitive = [transitive_mojopkgs] + [depset(transitive_libs)]),
            ).merge_all(transitive_runfiles),
        ),
        platform_common.TemplateVariableInfo({
            "COMPUTED_IMPORT_PATH": ",".join(sorted(sets.to_list(import_paths))),
            "COMPUTED_LIBS": ",".join(sorted(shared_libs)),
        }),
    ]

mojo_test_environment = rule(
    implementation = _mojo_test_environment_implementation,
    attrs = {
        "short_path": attr.bool(default = True),
        "data": attr.label_list(
            providers = [MojoInfo],
        ),
        "libs": attr.label_list(
            providers = [CcInfo],
        ),
    },
)
