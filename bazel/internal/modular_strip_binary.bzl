"""Strip a binary"""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@build_bazel_rules_android//:link_hack.bzl", "create_compile_variables_hack", "strip_opts_hack")  # See link_hack.bzl for details
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def register_strip_action(*, ctx, input_file, output_file):
    """Setup the action for stripping the given binary.

    Args:
        ctx: The rule context.
        input_file: The input binary to strip.
        output_file: The output stripped binary.
    """
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    strip_variables = create_compile_variables_hack(
        cc_toolchain = cc_toolchain,
        feature_configuration = feature_configuration,
        output_file = output_file.path,
        input_file = input_file.path,
        strip_opts = strip_opts_hack(ctx),
    )
    strip_arguments = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.strip,
        variables = strip_variables,
    )
    strip_env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.strip,
        variables = strip_variables,
    )

    ctx.actions.run(
        inputs = depset(
            direct = [input_file],
            transitive = [cc_toolchain.all_files],
        ),
        outputs = [output_file],
        use_default_shell_env = True,
        env = strip_env,
        executable = cc_common.get_tool_for_action(
            feature_configuration = feature_configuration,
            action_name = ACTION_NAMES.strip,
        ),
        toolchain = "@bazel_tools//tools/cpp:toolchain_type",
        progress_message = "Stripping %{label}",
        mnemonic = "CcStrip",
        arguments = strip_arguments,
    )

def _modular_strip_binary_impl(ctx):
    stripped_output = ctx.actions.declare_file(ctx.file.binary.basename + ".stripped/" + (ctx.attr.output_name or ctx.file.binary.basename))
    register_strip_action(
        ctx = ctx,
        input_file = ctx.file.binary,
        output_file = stripped_output,
    )

    return [DefaultInfo(files = depset([stripped_output]))]

modular_strip_binary = rule(
    implementation = _modular_strip_binary_impl,
    attrs = {
        "binary": attr.label(allow_single_file = True, mandatory = True),
        "output_name": attr.string(mandatory = False),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
