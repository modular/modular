"""A rule that builds a target with `--//:use_aiperf` forced on."""

def _use_aiperf_transition_impl(_settings, _attr):
    # The other dependency-group flags select conflicting versions of the
    # same packages in the `@multiple` pip aliases (see the uv `conflicts`
    # groups in bazel/pip/requirements/pyproject.toml), so leaving one on
    # alongside `use_aiperf` makes those aliases ambiguous. Force them off.
    return {
        "//:use_aiperf": True,
        "//:use_sglang": False,
        "//:use_vllm": False,
        "//:use_bench": False,
    }

use_aiperf_transition = transition(
    implementation = _use_aiperf_transition_impl,
    inputs = [],
    outputs = [
        "//:use_aiperf",
        "//:use_sglang",
        "//:use_vllm",
        "//:use_bench",
    ],
)

def _with_use_aiperf_impl(ctx):
    default = ctx.attr.dep[0][DefaultInfo]  # transitioned attrs are lists
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(output = executable, target_file = default.executable)
    providers = [DefaultInfo(
        executable = executable,
        files = depset([executable]),
        runfiles = default.default_runfiles,
    )]
    if RunEnvironmentInfo in ctx.attr.dep[0]:
        providers.append(ctx.attr.dep[0][RunEnvironmentInfo])
    return providers

with_use_aiperf = rule(
    doc = """Builds `dep` with `--//:use_aiperf` forced on and forwards it.

The flag is what selects aiperf's pinned versions in the `@multiple` pip
aliases, so only `dep`'s subgraph sees them. `dep`'s runfiles are forwarded
untouched, so its executable keeps appearing at its original runfiles path;
the executable itself is symlinked because a rule returning an executable
must create that file itself.
""",
    implementation = _with_use_aiperf_impl,
    executable = True,
    attrs = {
        "dep": attr.label(
            cfg = use_aiperf_transition,
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
