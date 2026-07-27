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

The rule reuses the ``interp_cache.bzl`` recipe: a ``py_binary``'s ``env`` block
carries the ``MODULAR_MOJO_MAX_*`` kernel-import vars only under
``bazel run``/``test``, never when exec'd as a build tool, so read the binary's
``RunEnvironmentInfo`` and re-inject it, then ``cd`` into the runfiles tree so
those short_path vars resolve.
"""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")

def _precompiled_mef_impl(ctx):
    # A single file output (per-target subdir keeps the basename <spec>.mef
    # unique across targets in the same package).
    mef = ctx.actions.declare_file(ctx.attr.name + "/" + ctx.attr.spec + ".mef")

    mojo_toolchain = ctx.toolchains["@rules_mojo//:toolchain_type"].mojo_toolchain_info

    cpu_target = None
    target = None
    for copt in mojo_toolchain.copts:
        if copt.startswith("--target-accelerator="):
            target = copt.removeprefix("--target-accelerator=")
        elif copt.startswith("--target-cpu="):
            cpu_target = copt.removeprefix("--target-cpu=")

    # not sure what this is about, is there a disconnect between the GC and Mojo?
    if target:
        target = target.replace("nvidia", "cuda")
        target = target.replace("amdgpu", "hip")

    binary = ctx.attr.producer[DefaultInfo].files_to_run
    env = dict(ctx.attr.producer[RunEnvironmentInfo].environment)

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(mef.path)
    args.add("--target")
    args.add(target)
    args.add("--cpu-target")
    args.add(cpu_target)
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
        env = env | {
            # If we import transformers but torch isn't available, we get a warning. Just silence it.
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        },
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
    },
    toolchains = [
        "@rules_mojo//:toolchain_type",
    ],
)

def precompiled_mefs(
        name,
        producer,
        specs,
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
        testonly: Whether the generated targets are test-only. Defaults to
            ``True`` (MEFs are test fixtures produced by a testonly producer).
        **kwargs: Common attrs (visibility, tags, ...) forwarded to all targets.
    """
    for spec in specs:
        _precompiled_mef(
            name = name + "_" + spec,
            producer = producer,
            spec = spec,
            testonly = testonly,
            **kwargs
        )
    native.filegroup(
        name = name,
        srcs = [name + "_" + spec for spec in specs],
        testonly = testonly,
        **kwargs
    )
