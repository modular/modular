"""Limit real exec config to when we actually need it.

This allows us to reduce duplicate building in the case that the host and
target are the same OS and architecture. This especially matters if you're
iterating on a target that is also used as a tool in the build.
"""

def _write_cfg(rctx, cfg, os, arch):
    rctx.file("cfg_workaround.bzl", content = """\
CFG_WORKAROUND = "{cfg}"
TARGET_CONSTRAINTS = ["@platforms//os:{os}", "@platforms//cpu:{arch}"]
""".format(
        cfg = cfg,
        os = os,
        arch = arch,
    ))

def _impl(rctx):
    rctx.file("BUILD.bazel", content = "")

    host_os = "macos" if "mac os x" in rctx.os.name else "linux"
    host_arch = "x86_64" if rctx.os.arch == "amd64" else "aarch64"
    target_os, target_arch = rctx.os.environ.get("MODULAR_TARGET", "{}:{}".format(host_os, host_arch)).split(":", 1)
    if host_os == "macos" and "MODULAR_REMOTE_EXEC" in rctx.os.environ:
        _write_cfg(rctx, "exec", target_os, target_arch)
        return

    if "MODULAR_TARGET" not in rctx.os.environ:
        _write_cfg(rctx, "target", target_os, target_arch)
        return

    if host_os == target_os and host_arch == target_arch:
        _write_cfg(rctx, "target", target_os, target_arch)
    else:
        _write_cfg(rctx, "exec", target_os, target_arch)

cfg_workaround = repository_rule(
    implementation = _impl,
    environ = [
        "MODULAR_REMOTE_EXEC",
        "MODULAR_TARGET",
    ],
)
