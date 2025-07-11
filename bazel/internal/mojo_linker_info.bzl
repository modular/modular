"""Fetch the flags that should be used when running Mojo in a test."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

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

def _mojo_linker_info_implementation(ctx):
    # NOTE: env should probably be used here but can't be passed through directly, right now it is only ZERO_AR_DATE
    linker_driver, system_libs, _, extra_files = _extract_linker_variables(ctx)

    return [
        DefaultInfo(
            runfiles = ctx.runfiles(transitive_files = extra_files),
        ),
        platform_common.TemplateVariableInfo({
            "MOJO_LINKER_DRIVER": linker_driver,
            "MOJO_LINKER_SYSTEM_LIBS": system_libs,
        }),
    ]

mojo_linker_info = rule(
    implementation = _mojo_linker_info_implementation,
    toolchains = use_cpp_toolchain() + [
        "@bazel_tools//tools/test:default_test_toolchain_type",
    ],
    fragments = ["cpp"],
)
