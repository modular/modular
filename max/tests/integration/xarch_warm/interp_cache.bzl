"""Build action that warms the eager-interpreter GC sweep into a warm store.

``warm_interp_cache`` runs a per-slot producer binary (``warm_cpu`` or
``warm_accelerators``) as a build action whose declared tree-artifact output
is a :class:`~max.engine.MefStore` directory: one artifact per warmed (family,
device slot) plus its manifest. A consumer test takes the directories as
``data`` and names them in ``MAX_EAGER_WARM_STORE`` (see ``eager_py_test.bzl``);
the store unions them, so nothing merges the fragments.

The rule reuses the ``//max/tests/integration/tools:precompile_pipeline``
recipe: a ``py_binary``'s ``env`` block (which carries the ``MODULAR_MOJO_MAX_*``
kernel-import vars) is only applied under ``bazel run``/``bazel test``, never
when the binary is exec'd as a ``genrule`` tool, so read the binary's
``RunEnvironmentInfo`` and re-inject it, then ``cd`` into the runfiles tree so
the short_path values in those vars resolve.
"""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")

def _warm_interp_cache_impl(ctx):
    store_dir = ctx.actions.declare_directory(ctx.attr.name + "_store")

    binary = ctx.attr.binary[DefaultInfo].files_to_run
    env = dict(ctx.attr.binary[RunEnvironmentInfo].environment)

    # No device hiding: warm_accelerators uses MAX's virtual-device knobs to
    # compile the GPU sweep with no physical GPU, so the worker's real
    # accelerators are irrelevant (warm_cpu sets none).

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(store_dir.path)

    # Optional --target: an empty target is a CPU-only warm (warm_cpu), the
    # producer then sets no virtual device and warms only the CPU slot.
    if ctx.attr.target:
        args.add("--target")
        args.add(ctx.attr.target)
    args.add("--cpu-target")
    args.add(ctx.attr.cpu_target)

    # Optional --family: warm only this GC family; empty warms all.
    if ctx.attr.family:
        args.add("--family")
        args.add(ctx.attr.family)

    # The binary's env vars (e.g. MODULAR_MOJO_MAX_IMPORT_PATH) hold short_path
    # values that resolve relative to the runfiles root, but a build action's
    # CWD is the execroot, so make the output path absolute up front, then cd
    # into the runfiles dir before running.
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
OUT="$PWD/$1"; shift
# The compiles' own cache is scratch: only the store is the action's output.
export MODULAR_DERIVED_PATH="$(mktemp -d)"
cd "${EXE}.runfiles/_main"
"$EXE" --out "$OUT" "$@"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        outputs = [store_dir],
        mnemonic = "WarmInterpCache",
        progress_message = "Warming eager interpreter GC cache %{output}",
    )

    return [DefaultInfo(files = depset([store_dir]))]

warm_interp_cache = rule(
    doc = "Runs the eager GC sweep as a build action; outputs a warm store dir.",
    implementation = _warm_interp_cache_impl,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            executable = True,
            # Deferred: the sweep uses only the graph compiler, never the op .so
            # kernels, but importing the sweep-builders runs
            # _interpreter_ops/__init__, which eagerly imports every op module,
            # so the producer builds op .so's it never uses. Splitting gc_sweeps
            # + gc_compile out of that init would drop the dep.
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
        "family": attr.string(
            doc = "Warm only this GC family, passed as --family (empty warms " +
                  "all). The BUILD loops _WARM_FAMILIES to make one fragment " +
                  "per family; the producer validates the name against " +
                  "GC_FAMILIES.",
        ),
    },
)
