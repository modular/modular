"""The Intel Mac toolchain repo. Modular does not ship one, so users build it."""

_ENV = "MODULAR_INTEL_MAC_TOOLCHAIN"

_MESSAGE = """\
The Intel Mac toolchain has not been built.

Intel Macs are community-supported, and Modular does not ship a toolchain for
them. To build one, run:

    bazel/internal/cc-toolchain/intel-mac/bootstrap.sh

Then add the line it prints to local.bazelrc. For details, see
Mojo/docs/contributing/building-on-intel-macs.md.
"""

# Placeholder files, so that analysis and queries work without a toolchain.
# These are the files that clang.BUILD and gperftools-macos.BUILD point at.
_PLACEHOLDER_TOOLS = [
    "clang",
    "clang++",
    "clang-format",
    "clang-tidy",
    "clangd",
    "dsymutil",
    "ld.lld",
    "lld",
    "llvm-ar",
    "llvm-as",
    "llvm-cov",
    "llvm-dwp",
    "llvm-install-name-tool",
    "llvm-nm",
    "llvm-objcopy",
    "llvm-objdump",
    "llvm-otool",
    "llvm-profdata",
    "llvm-ranlib",
    "llvm-strip",
]
# Files inside clang's resource directory, lib/clang/<major version>/.
_PLACEHOLDER_RESOURCE_FILES = [
    "include/README",
    "share/README",
    "lib/darwin/libclang_rt.asan_osx_dynamic.dylib",
    "lib/darwin/libclang_rt.tsan_osx_dynamic.dylib",
    "lib/darwin/libclang_rt.ubsan_osx_dynamic.dylib",
]

def _clang_versions(build):
    """The clang major versions a BUILD file names in lib/clang/<version>."""
    versions = []
    for part in build.split("lib/clang/")[1:]:
        version = part.split("/")[0].split('"')[0]
        if version.isdigit() and version not in versions:
            versions.append(version)
    return versions

def _impl(rctx):
    build = "\n".join([rctx.read(f) for f in rctx.attr.build_files])
    rctx.file("BUILD.bazel", build)
    path = rctx.getenv(_ENV, "")
    if path:
        # Relative paths are relative to the workspace.
        for child in rctx.workspace_root.get_child(path).readdir():
            rctx.symlink(child, child.basename)
        return

    script = "#!/bin/sh\ncat >&2 <<'EOF'\n{}EOF\nexit 1\n".format(_MESSAGE)
    for tool in _PLACEHOLDER_TOOLS:
        rctx.file("bin/" + tool, script, executable = True)
    # Follow the clang version in clang.BUILD, so a toolchain bump needs no
    # change here.
    for version in _clang_versions(build):
        for file in _PLACEHOLDER_RESOURCE_FILES:
            rctx.file("lib/clang/{}/{}".format(version, file), _MESSAGE)
    rctx.file("libtcmalloc_minimal.a", _MESSAGE)
    rctx.file("gperftools/tcmalloc.h", "".join(["// " + l.rstrip() + "\n" for l in _MESSAGE.splitlines()]) + '#error "The Intel Mac toolchain has not been built."\n')

intel_mac_toolchain = repository_rule(
    implementation = _impl,
    attrs = {
        "build_files": attr.label_list(doc = "BUILD files to join into the repo's BUILD.bazel."),
    },
    environ = [_ENV],
    doc = """The Intel Mac toolchain: clang, lld, LLVM tools and tcmalloc.

    If MODULAR_INTEL_MAC_TOOLCHAIN (--repo_env) names a directory, it is linked
    in. Without it, the repo holds placeholders that print how to build one.""",
)
