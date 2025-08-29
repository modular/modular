"""Run uv lock to generate a new uv.lock file."""

load("@rules_python//python:defs.bzl", "PyRuntimeInfo")

def _uv_lock_impl(ctx):
    executable = ctx.actions.declare_file(ctx.attr.name)
    lockfile_output = ctx.actions.declare_file(ctx.attr.name + "-new-uv.lock")

    py3_runtime = ctx.attr.py3_runtime[PyRuntimeInfo]

    ctx.actions.expand_template(
        template = ctx.file._template,
        output = executable,
        substitutions = {
            "{{existing_lockfile}}": ctx.file.existing_lockfile.short_path,
            "{{lockfile_output}}": lockfile_output.path,
            "{{pyproject}}": ctx.file.pyproject.short_path,
            "{{python}}": py3_runtime.interpreter.path,
            "{{uv}}": ctx.executable._uv.path,
        },
    )

    ctx.actions.run(
        executable = executable,
        outputs = [lockfile_output],
        inputs = [
            ctx.file.existing_lockfile,
            ctx.file.pyproject,
        ],
        tools = [
            ctx.executable._uv,
            py3_runtime.files,
            py3_runtime.interpreter,
        ],
        mnemonic = "UvLock",
        execution_requirements = {
            "no-sandbox": "1",  # Trust the uv cache
            "requires-network": "1",  # Allow network access for uv resolution
        },
    )

    return [
        DefaultInfo(files = depset([lockfile_output])),
    ]

uv_lock = rule(
    attrs = {
        "existing_lockfile": attr.label(mandatory = True, allow_single_file = True),
        "py3_runtime": attr.label(cfg = "exec", mandatory = True, doc = "The python3 runtime to use for uv resolution"),
        "pyproject": attr.label(mandatory = True, allow_single_file = True),
        "_template": attr.label(default = ":uv-lock.sh", allow_single_file = True),
        "_uv": attr.label(default = "//bazel/internal:uv", executable = True, cfg = "exec", allow_single_file = True),
    },
    implementation = _uv_lock_impl,
)
