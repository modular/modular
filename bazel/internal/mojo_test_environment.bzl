"""Fetch the environment variables that need to be set to execute Mojo during a test."""

load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_mojo//mojo:providers.bzl", "MojoInfo")
load("@rules_mojo//mojo/private:utils.bzl", "collect_mojoinfo")  # buildifier: disable=bzl-visibility
load("@rules_python//python:py_info.bzl", "PyInfo")

def _mojo_test_environment_implementation(ctx):
    mojo_toolchain = ctx.toolchains["@rules_mojo//:toolchain_type"].mojo_toolchain_info

    _, transitive_mojopkgs = collect_mojoinfo(ctx.attr.data)
    if not transitive_mojopkgs:
        return [
            CcInfo(),  # Requirement of py_test
            PyInfo(transitive_sources = depset()),  # Requirement of py_test
            platform_common.TemplateVariableInfo({
                "COMPILER_RT_PATH": "",
                "COMPUTED_IMPORT_PATH": "",
                "COMPUTED_LIBS": "",
                "MOJO_BINARY_PATH": "",
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
    transitive_files = []

    # TODO: This also contains runfiles, it probably should not.
    for tool in mojo_toolchain.all_tools:
        if type(tool) == type(depset()):
            transitive_files.append(tool)

    compilerrt = None
    for lib in mojo_toolchain.implicit_deps:
        if CcInfo not in lib:
            continue

        for linker_input in lib[CcInfo].linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                transitive_files.append(depset([library.dynamic_library]))

                if "CompilerRT" in lib.label.name:
                    compilerrt = library.dynamic_library

                path = library.dynamic_library.path
                if ctx.attr.short_path:
                    path = library.dynamic_library.short_path

                shared_libs.append(path)
                shared_libs.append("-Xlinker,-rpath,-Xlinker,{}".format(paths.dirname(path)))

    if not compilerrt:
        fail("CompilerRT library not found")

    return [
        CcInfo(),  # Requirement of py_test
        PyInfo(transitive_sources = depset()),  # Requirement of py_test
        DefaultInfo(
            runfiles = ctx.runfiles(
                transitive_files = depset(transitive = [transitive_mojopkgs] + transitive_files),
            ).merge_all(transitive_runfiles),
        ),
        platform_common.TemplateVariableInfo({
            "COMPUTED_IMPORT_PATH": ",".join(sorted(sets.to_list(import_paths))),
            "COMPUTED_LIBS": ",".join(sorted(shared_libs)),
            "MOJO_BINARY_PATH": mojo_toolchain.mojo.short_path if ctx.attr.short_path else mojo_toolchain.mojo.path,
            "COMPILER_RT_PATH": compilerrt.short_path if ctx.attr.short_path else compilerrt.path,
        }),
    ]

mojo_test_environment = rule(
    implementation = _mojo_test_environment_implementation,
    attrs = {
        "short_path": attr.bool(default = True),
        "data": attr.label_list(
            providers = [MojoInfo],
        ),
    },
    toolchains = ["@rules_mojo//:toolchain_type"],
)
