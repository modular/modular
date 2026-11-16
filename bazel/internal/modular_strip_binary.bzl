"""Strip a binary"""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//tools/build_defs/cc:link_hack.bzl", "create_compile_variables_hack", "strip_opts_hack")  # See link_hack.bzl for details

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

def _register_dsym_action(ctx):
    """Runs dsymutil on the binary, or returns None if dSYMs are disabled.

    The linker leaves the DWARF in the object files and records only their
    paths, relative to the execroot because of -oso_prefix, so this action can
    collect it as long as those objects are among its inputs.
    """
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    if not cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "generate_dsym_file"):
        return None

    # A superset of what the link used: this includes what the binary's
    # dynamic_deps linked too.
    objects = []
    for dep in ctx.attr.dsym_deps:
        for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                objects.extend(library.objects or [])
                objects.extend(library.pic_objects or [])
                if library.static_library:
                    objects.append(library.static_library)
                if library.pic_static_library:
                    objects.append(library.pic_static_library)

    dsym = ctx.actions.declare_directory(ctx.file.binary.basename + ".dSYM")
    ctx.actions.run(
        inputs = depset(direct = [ctx.file.binary] + objects),
        outputs = [dsym],
        executable = ctx.executable._dsymutil,
        arguments = ["-o", dsym.path, ctx.file.binary.path],
        progress_message = "Generating dSYM for %{label}",
        mnemonic = "Dsymutil",
    )
    return dsym

def _modular_strip_binary_impl(ctx):
    stripped_output = ctx.actions.declare_file(ctx.file.binary.basename + ".stripped/" + (ctx.attr.output_name or ctx.file.binary.basename))
    register_strip_action(
        ctx = ctx,
        input_file = ctx.file.binary,
        output_file = stripped_output,
    )

    debug_info = ctx.file.binary
    if ctx.attr.dsym_deps:
        debug_info = _register_dsym_action(ctx) or debug_info

    return [
        DefaultInfo(files = depset([stripped_output])),
        # Read by release_debug_info_aspect.
        OutputGroupInfo(modular_dwarf = depset([debug_info])),
    ]

modular_strip_binary = rule(
    implementation = _modular_strip_binary_impl,
    attrs = {
        "binary": attr.label(allow_single_file = True, mandatory = True),
        "output_name": attr.string(mandatory = False),
        "dsym_deps": attr.label_list(
            providers = [CcInfo],
            doc = "The libraries linked into `binary`, for its dSYM when " +
                  "--apple_generate_dsym is set. A cc_binary makes its own, " +
                  "but a cc_shared_library doesn't.",
        ),
        "_dsymutil": attr.label(
            default = "//bazel/internal/cc-toolchain/tools:dsymutil",
            cfg = "exec",
            executable = True,
        ),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
