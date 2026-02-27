"""Rule to pre-compile a model and output the cache directory."""

def modular_pipeline_target_name(spec):
    """Derive a unique target name from a PIPELINE_SPEC dict."""
    return "{}_{}".format(
        spec["pipeline"].lower().split("/")[1],
        spec["target"].replace(":", "_"),
    )

def _precompile_pipeline_impl(ctx):
    cache_dir = ctx.actions.declare_directory(ctx.attr.name + "_cache")

    binary = ctx.attr._binary[DefaultInfo].files_to_run

    env = dict(ctx.attr._binary[RunEnvironmentInfo].environment)
    env["HF_HUB_OFFLINE"] = "1"

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(cache_dir.path)
    args.add_all([
        "--pipeline",
        ctx.attr.pipeline,
        "--encoding",
        ctx.attr.encoding,
        "--devices",
        ctx.attr.devices,
        "--target",
        ctx.attr.target,
    ])

    # The binary's env vars (e.g. MODULAR_MOJO_MAX_IMPORT_PATH) contain
    # short_path values that resolve relative to the runfiles root. In a
    # build action, CWD is the execroot, so we cd into the runfiles
    # directory first. MODULAR_MAX_CACHE_DIR must be made absolute before
    # the cd so the output is written to the correct location.
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
export MODULAR_MAX_CACHE_DIR="$PWD/$1"; shift
cd "${EXE}.runfiles/_main"
"$EXE" "$@"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        outputs = [cache_dir],
    )

    return [DefaultInfo(files = depset([cache_dir]))]

modular_precompile_pipeline = rule(
    implementation = _precompile_pipeline_impl,
    attrs = {
        "pipeline": attr.string(mandatory = True),
        "encoding": attr.string(mandatory = True),
        "devices": attr.string(default = "gpu"),
        "target": attr.string(mandatory = True),
        "_binary": attr.label(
            default = "//max/tests/integration/tools:precompile_pipeline",
            executable = True,
            cfg = "exec",
        ),
    },
)
