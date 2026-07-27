"""Reusable build actions for precompiling test graphs to MEFs on CPU.

The GPU-time-saving pattern: a CPU-only build action runs a producer binary
(see ``test_common.mef_precompile.precompile_entrypoint``) under MAX's virtual-device
knobs to compile one graph to a MEF (no GPU needed -- just the target arch's
codegen from the toolchain), and a GPU test later consumes the produced MEFs to
initialize + execute. This moves the per-graph compile off the scarce GPU lane
onto cacheable CPU build actions.

Public API:

- ``precompiled_mefs(name, producer, specs, target, ...)``: the macro a test
  calls. Expands to one CPU build action per spec plus a ``filegroup``
  bundling the per-spec ``.mef`` files under a single ``name`` the consumer
  depends on.
- ``DEFAULT_CPU_TARGET``: a common preset for the host-CPU select so tests
  don't re-declare it.

The rule reuses the ``interp_cache.bzl`` recipe: a ``py_binary``'s ``env`` block
carries the ``MODULAR_MOJO_MAX_*`` kernel-import vars only under
``bazel run``/``test``, never when exec'd as a build tool, so read the binary's
``RunEnvironmentInfo`` and re-inject it, then ``cd`` into the runfiles tree so
those short_path vars resolve.
"""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")

# Host-CPU codegen target, sourced per build platform the same way the mojo
# toolchain is. Part of the MEF compile key, so it must match the consuming host.
DEFAULT_CPU_TARGET = select({
    "//:linux_x86_64": "triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3",
    "//:linux_aarch64": "triple=aarch64-unknown-linux-gnu;cpu=neoverse-n1",
})

def _precompiled_mef_impl(ctx):
    # A single file output (per-target subdir keeps the basename <spec>.mef
    # unique across targets in the same package).
    mef = ctx.actions.declare_file(ctx.attr.name + "/" + ctx.attr.spec + ".mef")

    binary = ctx.attr.producer[DefaultInfo].files_to_run
    env = dict(ctx.attr.producer[RunEnvironmentInfo].environment)

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(mef.path)
    args.add("--target")
    args.add(ctx.attr.target)
    args.add("--cpu-target")
    args.add(ctx.attr.cpu_target)
    args.add("--spec")
    args.add(ctx.attr.spec)

    # The binary's env vars hold short_path values that resolve relative to the
    # runfiles root, but a build action's CWD is the execroot: absolutize the
    # MEF output path up front, point MODULAR_DERIVED_PATH at a throwaway
    # scratch dir (MAX's compile caches land there rather than in a declared
    # output), then cd into the runfiles dir before running.
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
MEF_OUT="$PWD/$1"; shift
export MODULAR_DERIVED_PATH="$PWD/mef_scratch"
mkdir -p "$MODULAR_DERIVED_PATH"
cd "${EXE}.runfiles/_main"
"$EXE" --out "$MEF_OUT" "$@"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        outputs = [mef],
        mnemonic = "PrecompileMef",
        progress_message = "Precompiling MEF %{output}",
    )

    return [DefaultInfo(files = depset([mef]))]

_precompiled_mef = rule(
    doc = "Compiles one graph spec to a MEF as a CPU build action.",
    implementation = _precompiled_mef_impl,
    attrs = {
        "producer": attr.label(
            mandatory = True,
            executable = True,
            cfg = CFG_WORKAROUND,
            doc = "A modular_py_binary whose __main__ calls precompile_entrypoint().",
        ),
        "spec": attr.string(
            mandatory = True,
            doc = "Spec name to compile, passed as --spec.",
        ),
        "target": attr.string(
            mandatory = True,
            doc = "Virtual GPU target 'api:arch' (e.g. 'cuda:sm_100a').",
        ),
        "cpu_target": attr.string(
            mandatory = True,
            doc = "Host-CPU codegen descriptor, passed as --cpu-target.",
        ),
    },
)

def precompiled_mefs(
        name,
        producer,
        specs,
        target,
        cpu_target = DEFAULT_CPU_TARGET,
        target_compatible_with = None,
        testonly = True,
        **kwargs):
    """Compiles a list of graph specs to MEFs on CPU, bundled under one target.

    Expands to one CPU build action per spec (each stays an independent,
    cacheable action -- keeps compiles under the 900s action timeout) plus a
    ``filegroup`` collecting the per-spec ``<spec>.mef`` files, so the consumer
    test depends on a single ``:name`` and reads the files via
    ``$(rlocationpaths :name)`` (see :func:`~test_common.mef_precompile.mefs_from_env`).

    Args:
        name: Target name; the filegroup bundling the per-spec MEF files.
        producer: A modular_py_binary calling precompile_entrypoint() (label).
        specs: List of spec-name strings, one MEF per spec (``<spec>.mef``).
        target: Virtual GPU target 'api:arch' (e.g. "cuda:sm_100a").
        cpu_target: Host-CPU codegen descriptor. Defaults to DEFAULT_CPU_TARGET.
        target_compatible_with: Platform gate for the whole subgraph (e.g.
            ``["//:b200_gpu"]``). Applied to every generated target.
        testonly: Whether the generated targets are test-only. Defaults to
            ``True`` (MEFs are test fixtures produced by a testonly producer).
        **kwargs: Common attrs (visibility, tags, ...) forwarded to all targets.
    """
    for spec in specs:
        _precompiled_mef(
            name = name + "_" + spec,
            producer = producer,
            spec = spec,
            target = target,
            cpu_target = cpu_target,
            target_compatible_with = target_compatible_with,
            testonly = testonly,
            **kwargs
        )
    native.filegroup(
        name = name,
        srcs = [name + "_" + spec for spec in specs],
        target_compatible_with = target_compatible_with,
        testonly = testonly,
        **kwargs
    )
