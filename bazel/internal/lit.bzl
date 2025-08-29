"""Helpers for running lit tests in bazel"""

load("@rules_python//python/private:py_test_rule.bzl", upstream_py_test = "py_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "env_for_available_tools", "get_default_exec_properties", "get_default_test_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility
load(":mojo_test_environment.bzl", "mojo_test_environment")  # buildifier: disable=bzl-visibility

_HEADER_PATH_ADDITIONS = """
from python.runfiles import runfiles

_R = runfiles.Create()

for path in [x for x in "{}".split(":") if x]:
    if os.path.exists(path):
        tool_path = os.path.dirname(os.path.abspath(path))
    else:
        tool_path = os.path.dirname(_R.Rlocation(path))
    os.environ["PATH"] = tool_path + os.pathsep + os.environ["PATH"]
"""

_STRIP_OPTION = "//command_line_option:strip"

def _strip_transition_impl(settings, attr):
    output = {_STRIP_OPTION: settings[_STRIP_OPTION]}
    if attr.force_strip:
        output[_STRIP_OPTION] = "always"
    return output

_strip_transition = transition(
    implementation = _strip_transition_impl,
    inputs = [_STRIP_OPTION],
    outputs = [_STRIP_OPTION],
)

py_test = rule(
    implementation = lambda ctx: ctx.super(),
    parent = upstream_py_test,
    attrs = {
        "force_strip": attr.bool(mandatory = True),
    },
    cfg = _strip_transition,
)

# TODO: Replace with upstream passing 'py_test =' once we bump LLVM
def _lit_test(name, srcs, args = None, data = None, deps = None, **kwargs):
    args = args or []
    data = data or []
    deps = deps or []
    py_test(
        name = name,
        srcs = [Label("@llvm-project//llvm:lit"), "//bazel/internal/llvm-lit:lit_shim.py"],
        main = Label("//bazel/internal/llvm-lit:lit_shim.py"),
        args = args + ["-v"] + ["$(execpath %s)" % src for src in srcs],
        data = data + srcs,
        deps = deps + [Label("@llvm-project//llvm:lit")],
        **kwargs
    )

def _generate_site_cfg_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.out)
    tools_loader = ctx.actions.declare_file(ctx.label.name + ".tools_loader")
    combined_cfg = ctx.actions.declare_file(ctx.label.name + ".combined_cfg")

    expanded_locations = {}
    for key, value in ctx.attr.custom_substitutions.items():
        expanded_locations[key] = ctx.expand_location(
            value,
            targets = ctx.attr.data,
        )

    expanded_features = []
    for feature in ctx.attr.available_features:
        expanded_features.append(
            ctx.expand_make_variables("available_features", feature, {}),
        )

    ctx.actions.expand_template(
        template = ctx.file._load_bazel_tools,
        output = tools_loader,
        substitutions = {
            "{custom_substitutions}": json.encode(expanded_locations),
            "{available_features}": json.encode(expanded_features),
        },
    )

    ctx.actions.run_shell(
        command = "set -eu; cat $1 $2 > {}".format(combined_cfg.path),
        arguments = [ctx.file.src.path, tools_loader.path],
        inputs = [ctx.file.src, tools_loader],
        outputs = [combined_cfg],
    )

    ctx.actions.expand_template(
        template = combined_cfg,
        output = output,
        substitutions = {
            "@LIT_SITE_CFG_IN_HEADER@": _HEADER_PATH_ADDITIONS.format(
                ctx.expand_make_variables(
                    "path",
                    ctx.expand_location(
                        ctx.attr.path,
                        targets = ctx.attr.data,
                    ),
                    {},
                ),
            ),
        },
    )

    return [DefaultInfo(files = depset([output]))]

_generate_site_cfg = rule(
    implementation = _generate_site_cfg_impl,
    attrs = {
        "data": attr.label_list(
            mandatory = True,
            cfg = "target",  # NOTE: this is the key difference vs a standard genrule
            doc = "Data dependencies of the lit cfg.",
            allow_files = True,
        ),
        "out": attr.string(
            mandatory = True,
            doc = "the output file name",
        ),
        "path": attr.string(
            mandatory = True,
            doc = "PATH replacements",
        ),
        "src": attr.label(
            mandatory = True,
            doc = "The input lit cfg",
            allow_single_file = True,
        ),
        "custom_substitutions": attr.string_dict(
            mandatory = True,
            doc = "A dictionary of replacement key to string value for custom lit replacements, subject to location expansion",
        ),
        "available_features": attr.string_list(
            mandatory = True,
            doc = "Lit features to always enable",
        ),
        "_load_bazel_tools": attr.label(
            default = Label(":load_bazel_tools.py"),
            allow_single_file = True,
        ),
    },
)

def lit_tests(
        name,
        srcs,
        tools = [],
        deps = [],
        mojo_deps = [],
        data = [],
        custom_substitutions = {},
        tags = [],
        env = {},
        env_inherit = [],
        target_compatible_with = [],
        gpu_constraints = [],
        generate_litcfg = True,
        unique_suffix = None,
        size = None,
        force_strip = False,
        exec_properties = {}):
    """Create test rules for all lit tests in the current directory.

    Args:
        name: The name of the underlying test_suite to run all the created tests
        srcs: The files that should be treated as lit tests
        tools: The tools being exercised in the tests, used as replacements
        deps: Any py_library deps to add to the lit test target
        mojo_deps: Mojo library dependencies required by the tests, adding anything here automatically adds mojo to the environment
        data: Any files required at test runtime
        custom_substitutions: Custom lit substitutions, use sparingly.
        tags: Tags to set on the underlying targets
        env: Environment variables to launch the tests with
        env_inherit: Environment variables to launch the tests with from the global env
        target_compatible_with: https://bazel.build/extending/platforms#skipping-incompatible-targets
        gpu_constraints: GPU requirements for the tests
        generate_litcfg: Generate the lit.cfg.py needed to run the test
        unique_suffix: Optional suffix to add to individual test target names
        size: Test size: https://bazel.build/reference/test-encyclopedia
        force_strip: Whether binaries should be stripped via a transition, do not use, just here to workaround a bug
        exec_properties: Remote exec resources https://www.buildbuddy.io/docs/rbe-platforms/#runner-resource-allocation
    """

    validate_gpu_tags(tags, gpu_constraints)

    tool_names = []
    tool_paths = []

    for tool in tools:
        separator = "/"
        if ":" in tool:
            separator = ":"
        tool_names.append('"{}"'.format(tool.split(separator)[-1]))
        tool_paths.append('"{}"'.format(Label(tool).package))

    tools = list(tools)
    tools.append("@llvm-project//llvm:llvm-symbolizer")

    litcfg = "lit.cfg.py"
    local_litcfg = "lit.site.cfg.py.in"
    if generate_litcfg:
        extensions = sorted(set([".{}".format(x.rsplit(".", 1)[-1]) for x in srcs]))
        extensions_string = ", ".join(['"{}"'.format(ext) for ext in extensions])
        litcfg = name + ".cfg.py"
        local_litcfg = name + ".site.cfg.py.in"
        native.genrule(
            name = name + "_gen_lit_cfg",
            outs = [name + ".cfg.py"],
            cmd = """
cat > $(OUTS) <<EOF
config.name = "//{package}"
config.suffixes = [{extensions_string}]
EOF
            """.format(
                package = native.package_name(),
                extensions_string = extensions_string,
            ),
            testonly = True,
        )
        native.genrule(
            name = name + "_gen_lit_site_cfg",
            outs = [name + ".site.cfg.py.in"],
            cmd = """
cat > $(OUTS) <<EOF
@LIT_SITE_CFG_IN_HEADER@

lit_config.load_config(
    config, "./bazel/internal/llvm-lit/lit.common.configured"
)

lit_config.load_config(config, "{package}/{name}.cfg.py")
EOF
            """.format(
                package = native.package_name(),
                name = name,
            ),
            testonly = True,
        )

    mojo_test_deps = []
    mojo_test_env = {}
    extra_tool_paths = []
    if mojo_deps:
        mojo_test_deps.append(name + "_mojo_deps")
        mojo_test_env = {
            "MODULAR_MOJO_MAX_COMPILERRT_PATH": "$(COMPILER_RT_PATH)",
            "MODULAR_MOJO_MAX_DRIVER_PATH": "$(MOJO_BINARY_PATH)",
            "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
            "MODULAR_MOJO_MAX_LINKER_DRIVER": "$(MOJO_LINKER_DRIVER)",
            "MODULAR_MOJO_MAX_LLD_PATH": "$(LLD_PATH)",
            "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
            "MODULAR_MOJO_MAX_SYSTEM_LIBS": "$(MOJO_LINKER_SYSTEM_LIBS)",
        }
        extra_tool_paths.append("$(MOJO_BINARY_PATH)")
        mojo_test_environment(
            name = name + "_mojo_deps",
            data = mojo_deps,
            target_compatible_with = target_compatible_with,
            testonly = True,
        )

    _generate_site_cfg(
        name = name + "_lit_site_cfg_py",
        src = local_litcfg,
        out = name + ".site.cfg.py",
        data = mojo_deps + data + tools,
        custom_substitutions = custom_substitutions,
        testonly = True,
        tags = tags,
        path = ":".join(["$(rlocationpath {})".format(tool) for tool in tools] + extra_tool_paths),
        target_compatible_with = target_compatible_with,
        available_features = ["$(GPU_LIT_FEATURE)", "$(GPU_BRAND_LIT_FEATURE)"],
        toolchains = ["//bazel/internal:current_gpu_toolchain"] + mojo_test_deps,
    )

    default_env = {
        "LIT_PRESERVES_TMP": "1",
        "MODULAR_LIT_TEST": "1",
        "ZERO_AR_DATE": "1",
    } | GPU_TEST_ENV | get_default_test_env(exec_properties)

    extra_data = [
        "//bazel/internal:asan-suppressions.txt",
        "//bazel/internal:lsan-suppressions.txt",
    ]
    default_args = ["--config-prefix=" + name]
    kwargs = {
        "deps": deps + mojo_test_deps + [
            "//bazel/internal/llvm-lit:modular_test_format",
            "@rules_python//python/runfiles",
        ],
        "toolchains": mojo_test_deps + [
            "//bazel/internal:current_gpu_toolchain",
            "//bazel/internal:lib_toolchain",
        ],
        "data": data + tools + extra_data + mojo_test_deps + [
            litcfg,
            ":{}_lit_site_cfg_py".format(name),
            "//bazel/internal/llvm-lit:lit_data",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:not",
        ],
        "env": env_for_available_tools() | default_env | env | mojo_test_env,
        "env_inherit": env_inherit,
        "size": size,
        "exec_properties": get_default_exec_properties(tags, gpu_constraints) | exec_properties,
        "force_strip": force_strip,
    }

    for src in srcs:
        _lit_test(
            name = "%s%s.test" % (src.replace("🔥", "fire"), ("." + unique_suffix) if unique_suffix else ""),
            srcs = [src],
            target_compatible_with = target_compatible_with + gpu_constraints,
            tags = tags + ["no-mypy", "lit"],
            args = default_args + ["-a"],
            **kwargs
        )

    upstream_py_test(
        name = name + ".validate_lit_features",
        data = srcs + extra_data,
        srcs = ["//bazel/internal/llvm-lit:validate_lit_features.py"],
        main = "//bazel/internal/llvm-lit:validate_lit_features.py",
        args = [native.package_name()] + srcs,
        env = default_env | {"RUNS_ON_GPU": str("gpu" in tags)},
        toolchains = [
            "//bazel/internal:current_gpu_toolchain",
        ],
        tags = [
            "no-mypy",
            "validate-lit-features",
            "lint-test",
        ],
        timeout = "short",
    )

    # Create a test suite that runs all the tests which is faster for large test suites
    _lit_test(
        name = name,
        srcs = srcs,
        tags = tags + ["manual", "no-mypy"],
        target_compatible_with = target_compatible_with + gpu_constraints,
        args = default_args + ["-sv"],
        **kwargs
    )
