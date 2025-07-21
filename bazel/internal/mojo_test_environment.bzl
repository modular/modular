"""Fetch the environment variables that need to be set to execute Mojo during a test."""

load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_mojo//mojo:providers.bzl", "MojoInfo")
load("@rules_mojo//mojo/private:utils.bzl", "collect_mojoinfo")  # buildifier: disable=bzl-visibility
load("@rules_python//python:py_info.bzl", "PyInfo")

def _extract_linker_variables(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    linker_driver = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_executable,
    )
    variables = cc_common.create_link_variables(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
    )
    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_executable,
        variables = variables,
    )

    link_arguments = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_link_executable,
        variables = variables,
    )

    # TODO: Fix -Wl, exclusion
    system_libs = ",".join([x for x in link_arguments if not x.startswith("-Wl,")])

    return linker_driver, system_libs, env, cc_toolchain.all_files

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
                "LLD_PATH": "",
                "MOJO_BINARY_PATH": "",
                "MOJO_LINKER_DRIVER": "",
                "MOJO_LINKER_SYSTEM_LIBS": "",
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
    transitive_files = [depset([mojo_toolchain.lld])]

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

    # NOTE: env should probably be used here but can't be passed through directly, right now it is only ZERO_AR_DATE
    linker_driver, system_libs, _, extra_files = _extract_linker_variables(ctx)

    return [
        CcInfo(),  # Requirement of py_test
        PyInfo(transitive_sources = depset()),  # Requirement of py_test
        DefaultInfo(
            runfiles = ctx.runfiles(
                transitive_files = depset(transitive = [transitive_mojopkgs] + transitive_files + [extra_files]),
            ).merge_all(transitive_runfiles),
        ),
        platform_common.TemplateVariableInfo({
            "COMPILER_RT_PATH": compilerrt.short_path if ctx.attr.short_path else compilerrt.path,
            "COMPUTED_IMPORT_PATH": ",".join(sorted(sets.to_list(import_paths))),
            "COMPUTED_LIBS": ",".join(sorted(shared_libs)),
            "LLD_PATH": mojo_toolchain.lld.short_path if ctx.attr.short_path else mojo_toolchain.lld.path,
            "MOJO_BINARY_PATH": mojo_toolchain.mojo.short_path if ctx.attr.short_path else mojo_toolchain.mojo.path,
            "MOJO_LINKER_DRIVER": linker_driver,
            "MOJO_LINKER_SYSTEM_LIBS": system_libs,
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
    toolchains = use_cpp_toolchain() + [
        "@bazel_tools//tools/test:default_test_toolchain_type",
        "@rules_mojo//:toolchain_type",
    ],
    fragments = ["cpp"],
)
