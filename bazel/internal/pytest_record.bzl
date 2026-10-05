"""Runs a pytest target as a build action, to record what its graphs compile to.

Compiling a graph needs the target arch, not the accelerator, but executing one
needs the accelerator: a GPU test that compiles its own graphs therefore spends
most of a scarce GPU worker's time on work a CPU could have done. This rule runs
the test's own pytest invocation on a CPU worker under virtual devices, where
every compile writes its artifact and the run stops at the first thing that
needs real hardware. The test then runs on the GPU with those artifacts in its
runfiles and initializes them instead of compiling.

The action is the same program as the test, from the same revision, differing
only in the environment the plugin
(``oss/modular/bazel/internal/precompile_mefs_plugin.py``) reads. That is what
makes the two runs comparable: what identifies an artifact is which test or
fixture compiled it and in what order, so anything that changes what the test
does changes the artifacts too, and the plugin reports the divergence rather
than silently recompiling.

Verdicts belong to the GPU run, so this action exits 0 once it has written its
manifest and its record of where each test stopped. Only a run that could not
produce those -- a collection error, a usage error -- fails the build.
"""

load(":mojo_targets.bzl", "mojo_targets_from_toolchain")

# `pytest_runner.py` expects the environment `bazel test` gives it, and a build
# action gives it none of that, so the wrapper supplies it. Paths that have to
# be absolute are absolutized here, before the `cd` into the runfiles tree that
# the binary's `short_path` env vars resolve against.
_RECORD_SH = """\
set -eu
EXE="$PWD/$1"; shift
OUT="$PWD/$1"; shift
XML="$PWD/$1"; shift
SCRATCH="$OUT.scratch"
mkdir -p "$OUT" "$SCRATCH/tmp" "$SCRATCH/derived"
export MODULAR_RUNNING_TESTS=1
export TEST_TMPDIR="$SCRATCH/tmp"
export TEST_SHARD_STATUS_FILE="$SCRATCH/shard_status"
export XML_OUTPUT_FILE="$XML"
export MODULAR_DERIVED_PATH="$SCRATCH/derived"
export PRECOMPILE_MEFS_RECORD_DIR="$OUT"
cd "${EXE}.runfiles/_main"
# A whole pytest transcript per shard is more than a build log should carry,
# and bazel prints whatever an action writes -- including the header naming
# the action -- so a shard that recorded what it was asked to says nothing at
# all. A failure hands over the transcript. What each shard recorded is in
# the junit XML the rule declares and in `record.json` beside the artifacts.
if ! "$EXE" "$@" > "$SCRATCH/record.log" 2>&1; then
    cat "$SCRATCH/record.log" >&2
    exit 1
fi
rm -rf "$SCRATCH"
"""

def _pytest_record_impl(ctx):
    targets = mojo_targets_from_toolchain(ctx)

    # `modular_py_test` only emits this rule for a test that declares
    # gpu_constraints and excludes Apple, so neither of these is reachable
    # from the attribute; they catch a hand-written target instead.
    if not targets.accelerator:
        fail(
            "the mojo toolchain names no target accelerator, so there is " +
            "nothing to record for; gate this target with gpu_constraints",
        )
    if targets.accelerator.startswith("metal:"):
        fail(
            "Metal has no virtual compilation device (MOCO-2411), so nothing " +
            "can compile for it without the GPU attached; exclude //:apple_gpu",
        )

    binary = ctx.attr.producer[DefaultInfo].files_to_run

    # A py_binary carries the `MODULAR_MOJO_MAX_*` kernel-import vars in its
    # `env` block, which only `bazel run`/`test` applies: read them off the
    # binary and re-inject them here.
    env = {
        key: value
        for key, value in ctx.attr.producer[RunEnvironmentInfo].environment.items()
        if key not in ctx.attr.strip_env
    }
    pytest_args = [
        ctx.expand_location(arg, ctx.attr.expansion_targets)
        for arg in ctx.attr.pytest_args
    ]

    outs = []
    xmls = []
    for index in range(ctx.attr.shards):
        out = ctx.actions.declare_directory("{}.shard{}".format(ctx.label.name, index))
        xml = ctx.actions.declare_file("{}.shard{}.xml".format(ctx.label.name, index))
        outs.append(out)
        xmls.append(xml)

        args = ctx.actions.args()
        args.add(binary.executable)
        args.add(out.path)
        args.add(xml.path)
        args.add_all(pytest_args)

        # Through the environment rather than the command line, so
        # `pytest_runner.py` shards exactly as it does under `bazel test`.
        shard_env = {
            "TEST_TOTAL_SHARDS": str(ctx.attr.shards),
            "TEST_SHARD_INDEX": str(index),
        } if ctx.attr.shards > 1 else {}

        ctx.actions.run_shell(
            command = _RECORD_SH,
            arguments = [args],
            tools = [binary],
            use_default_shell_env = True,
            outputs = [out, xml],
            env = env | {
                "PRECOMPILE_MEFS_MODE": "record",
                "PRECOMPILE_MEFS_ACCELERATOR": targets.accelerator,
                "PRECOMPILE_MEFS_CPU_TARGET": targets.cpu_target,
                "TEST_TARGET": ctx.attr.test_label,
                # A build action has no network, and the executors mount the
                # shared HuggingFace cache read-only, so a test that resolves a
                # model resolves it offline or not at all.
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
                # The worker may hold accelerators this record run must not
                # find: the whole point is to compile as if it had none.
                "CUDA_VISIBLE_DEVICES": "",
                "HIP_VISIBLE_DEVICES": "",
            } | shard_env | ctx.attr.env,
            mnemonic = "PrecompileMefs",
            progress_message = "Recording MEFs for %{label} shard " +
                               str(index + 1) + " of " + str(ctx.attr.shards),
        )

    return [
        DefaultInfo(files = depset(outs)),
        OutputGroupInfo(junit = depset(xmls)),
    ]

pytest_record = rule(
    doc = "Records the MEFs a pytest target's graphs compile to, on CPU.",
    implementation = _pytest_record_impl,
    attrs = {
        "env": attr.string_dict(
            doc = "Extra environment for the record run, applied last.",
        ),
        "expansion_targets": attr.label_list(
            allow_files = True,
            doc = "Targets `$(location ...)` in `pytest_args` may name.",
        ),
        "producer": attr.label(
            mandatory = True,
            executable = True,
            # This binary runs during the build, which is what the exec
            # configuration is for. It also decides which flavor of an
            # accelerator-specific wheel the producer gets: in the exec
            # configuration the lane is the CPU builder the action runs on, so
            # torch resolves to the +cpu wheel that imports there rather than
            # the lane's CUDA or ROCm one, whose runtime the builder does not
            # have. What the artifact is compiled *for* does not come from here
            # -- the rule reads the accelerator and host-CPU targets off its own
            # toolchain, in the target configuration.
            cfg = "exec",
            doc = "A py_binary running the test's pytest invocation.",
        ),
        # Not `args`: that name is reserved on rules with an executable.
        "pytest_args": attr.string_list(
            doc = "The arguments the consumer test passes to pytest.",
        ),
        "shards": attr.int(
            default = 1,
            doc = "How many record actions to split the test across.",
        ),
        "strip_env": attr.string_list(
            doc = "Variables to drop from the producer's own environment.",
        ),
        "test_label": attr.string(
            doc = "The consumer test's label, which pytest names its suite after.",
        ),
    },
    toolchains = [
        "@rules_mojo//:toolchain_type",
    ],
)
