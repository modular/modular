"""Build actions that warm the eager-interpreter GC sweep into a cache dir.

Two rules:

- ``warm_interp_cache`` runs a per-slot producer binary (``warm_cpu`` or
  ``warm_accelerators``) as a build action with ``MODULAR_DERIVED_PATH`` pointed
  at a declared tree-artifact output; the sweep exports one MEF per warmed
  device slot + a manifest + a warm stamp.
- ``compose_warm_cache`` merges a ``warm_cpu`` output dir and a
  ``warm_accelerators`` output dir into one GPU-consumer cache dir (equivalent
  in shape to the old unified warm), so the CPU sweep is a single lane-shared
  action rather than baked into each lane-keyed GPU action.

A consumer test depends on the produced directory via ``data`` and points the
same ``MODULAR_DERIVED_PATH`` at it, the real build-action-output ->
test-input dependency edge, giving CI a warmed cache produced once at build
time.

Both rules reuse the ``//max/tests/integration/tools:precompile_pipeline``
recipe: a ``py_binary``'s ``env`` block (which carries the ``MODULAR_MOJO_MAX_*``
kernel-import vars) is only applied under ``bazel run``/``bazel test``, never
when the binary is exec'd as a ``genrule`` tool, so read the binary's
``RunEnvironmentInfo`` and re-inject it, then ``cd`` into the runfiles tree so
the short_path values in those vars resolve.
"""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")

def _warm_interp_cache_impl(ctx):
    derived_dir = ctx.actions.declare_directory(ctx.attr.name + "_derived")

    binary = ctx.attr.binary[DefaultInfo].files_to_run
    env = dict(ctx.attr.binary[RunEnvironmentInfo].environment)

    # No device hiding: warm_accelerators uses MAX's virtual-device knobs to
    # compile the GPU sweep with no physical GPU, so the worker's real
    # accelerators are irrelevant (warm_cpu sets none).

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(derived_dir.path)

    # Optional --target: an empty target is a CPU-only warm (warm_cpu), the
    # producer then sets no virtual device, warms only the CPU slot, and writes
    # a manifest with no "gpu" key.
    if ctx.attr.target:
        args.add("--target")
        args.add(ctx.attr.target)
    args.add("--cpu-target")
    args.add(ctx.attr.cpu_target)

    # The binary's env vars (e.g. MODULAR_MOJO_MAX_IMPORT_PATH) hold short_path
    # values that resolve relative to the runfiles root, but a build action's
    # CWD is the execroot, so make MODULAR_DERIVED_PATH absolute up front,
    # then cd into the runfiles dir before running.
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
export MODULAR_DERIVED_PATH="$PWD/$1"; shift
mkdir -p "$MODULAR_DERIVED_PATH"
cd "${EXE}.runfiles/_main"
"$EXE" "$@"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        outputs = [derived_dir],
        mnemonic = "WarmInterpCache",
        progress_message = "Warming eager interpreter GC cache %{output}",
    )

    return [DefaultInfo(files = depset([derived_dir]))]

warm_interp_cache = rule(
    doc = "Runs the eager GC sweep as a build action; outputs a warmed cache dir.",
    implementation = _warm_interp_cache_impl,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            executable = True,
            # CFG_WORKAROUND, not a literal "exec": the producer runs during the
            # build, but _interpreter_ops is ALSO a target-config dep of the
            # consumer test, so a hardcoded "exec" compiles that heavy mojo lib
            # twice (once per config). CFG_WORKAROUND resolves to "target" when
            # host os+arch == target (dedup, one shared build) and only to a
            # real "exec" on a genuine cross-compile. Force-load adopts by path
            # and bypasses the toolchain cache key regardless of config, so
            # adoption stays correct either way; the receipt's force-load marker
            # (not a keyed cache hit) is what proves the bypass, so the proof
            # holds even when producer and consumer share a config, and the true
            # exec/target mismatch is still exercised on a cross-compile.
            #
            # Deferred path with more juice: the sweep uses only the
            # graph compiler, never the op .so kernels, but importing the
            # sweep-builders runs _interpreter_ops/__init__, which eagerly
            # imports every op module, so the producer builds op .so's it
            # never uses. Splitting the gc_sweeps and gc_compile out of that
            # init would drop the dep. Payoff shrinks over time, though: as
            # more ops move to the graph compiler, the op .so's (and their
            # build time) get smaller.
            cfg = CFG_WORKAROUND,
            doc = "The per-slot producer modular_py_binary to run " +
                  "(warm_cpu or warm_accelerators).",
        ),
        "target": attr.string(
            doc = "Virtual GPU target 'api:arch' (e.g. 'cuda:sm_100a'), " +
                  "passed to the producer as --target. Empty (the default) warms " +
                  "CPU-only: no --target is passed and only the CPU slot is warmed.",
        ),
        "cpu_target": attr.string(
            mandatory = True,
            doc = "Virtual host-CPU target descriptor (e.g. " +
                  "'triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3'), passed to " +
                  "the producer as --cpu-target. Sourced from a select() " +
                  "mirroring the mojo toolchain's per-platform target.",
        ),
    },
)

def _compose_warm_cache_impl(ctx):
    out_dir = ctx.actions.declare_directory(ctx.attr.name + "_derived")

    binary = ctx.attr.binary[DefaultInfo].files_to_run
    env = dict(ctx.attr.binary[RunEnvironmentInfo].environment)

    # warm_cpu / warm_accelerators each produce a single tree-artifact dir.
    cpu_dir = ctx.attr.cpu_warm[DefaultInfo].files.to_list()[0]
    accel_dir = ctx.attr.accelerators_warm[DefaultInfo].files.to_list()[0]

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(out_dir.path)
    args.add(cpu_dir.path)
    args.add(accel_dir.path)

    # Same absolutize-then-cd dance as warm_interp_cache: a build action's CWD
    # is the execroot, but the binary's env short_path values resolve relative
    # to its runfiles root, so make every path absolute before cd'ing in.
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
export MODULAR_DERIVED_PATH="$PWD/$1"; shift
CPU_WARM="$PWD/$1"; shift
ACCEL_WARM="$PWD/$1"; shift
mkdir -p "$MODULAR_DERIVED_PATH"
cd "${EXE}.runfiles/_main"
"$EXE" --cpu-warm "$CPU_WARM" --accelerators-warm "$ACCEL_WARM"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        inputs = [cpu_dir, accel_dir],
        outputs = [out_dir],
        mnemonic = "ComposeInterpWarmCache",
        progress_message = "Composing eager interpreter GC cache %{output}",
    )

    return [DefaultInfo(files = depset([out_dir]))]

compose_warm_cache = rule(
    doc = "Merges a warm_cpu output + a warm_accelerators output into one " +
          "GPU-consumer cache dir (union of MEFs + merged manifest + the " +
          "GPU-context stamp).",
    implementation = _compose_warm_cache_impl,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            executable = True,
            # See the CFG_WORKAROUND note on warm_interp_cache. compose is pure
            # stdlib (no _interpreter_ops), so there's no double-build to dedup
            # here, kept consistent with warm_interp_cache.
            cfg = CFG_WORKAROUND,
            doc = "The compose modular_py_binary to run.",
        ),
        "cpu_warm": attr.label(
            mandatory = True,
            doc = "The warm_cpu tree-artifact dir (CPU MEFs + CPU manifest).",
        ),
        "accelerators_warm": attr.label(
            mandatory = True,
            doc = "The warm_accelerators tree-artifact dir (accelerator MEFs " +
                  "+ manifest fragment + GPU-context stamp).",
        ),
    },
)
