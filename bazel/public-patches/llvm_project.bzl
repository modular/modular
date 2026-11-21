"""Module extension to configure LLVM"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@llvm-raw//utils/bazel:configure.bzl", _llvm_configure = "llvm_configure")

# Mirrors the pyyaml repo that upstream's own llvm_repos_extension defines, so
# that @pyyaml//:yaml resolves from the generated llvm-project repo.
_PYYAML_CONTENT = """\
load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for PyYAML)
    licenses = ["notice"],
)

py_library(
    name = "yaml",
    srcs = glob(["yaml/*.py"]),
)
"""

# Stands in for upstream's linux_uapi_setup, which downloads a Debian package
# (llvm/llvm-project#216868) that our downloader config blocks. Only label
# resolution needs this repo, never its headers.
def _linux_uapi_stub_impl(repository_ctx):
    repository_ctx.file("BUILD.bazel", """\
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "linux_uapi_headers",
    visibility = ["//visibility:public"],
)
""")

_linux_uapi_stub = repository_rule(implementation = _linux_uapi_stub_impl)

BACKENDS = [
    "AArch64",
    "RISCV",
    "X86",
]

def _llvm_project_impl(module_ctx):
    # Check the raw names before they collapse into dictionaries: a repeated or
    # overlapping name is a configuration mistake, not something to merge.
    targets = {t: None for t in BACKENDS}
    optional_targets = {}
    for mod in module_ctx.modules:
        for tag in mod.tags.configure:
            for t in tag.extra_targets:
                if t in targets:
                    fail("LLVM target {} is listed more than once in extra_targets or is a default backend".format(t))
                targets[t] = None
            for t in tag.optional_targets:
                if t in optional_targets:
                    fail("LLVM target {} is listed more than once in optional_targets".format(t))
                optional_targets[t] = None
    for t in optional_targets:
        if t in targets:
            fail("LLVM target {} is listed both as builtin and in optional_targets".format(t))

    # llvm-libc's build rules reference @linux_uapi and @pyyaml. Upstream
    # supplies both from its own llvm_repos_extension, which we do not use, so
    # define them here or label resolution inside the generated llvm-project
    # repo fails. We never build llvm-libc, but Bazel still has to resolve the
    # labels: libc:hdrgen entered the transitive closure in this LLVM bump
    # (llvm/llvm-project#218992, #218993), and the @linux_uapi reference sits in
    # a select() branch, which Bazel resolves whether or not it is selected.
    _linux_uapi_stub(name = "linux_uapi")
    http_archive(
        name = "pyyaml",
        build_file_content = _PYYAML_CONTENT,
        sha256 = "f0a35d7f282a6d6b1a4f3f3965ef5c124e30ed27a0088efb97c0977268fd671f",
        strip_prefix = "pyyaml-5.1/lib3",
        url = "https://github.com/yaml/pyyaml/archive/refs/tags/5.1.zip",
    )

    _llvm_configure(
        name = "llvm-project",
        targets = sorted(targets.keys()),
        optional_targets = sorted(optional_targets.keys()),
    )

    return module_ctx.extension_metadata(reproducible = True)

_configure = tag_class(
    attrs = {
        "extra_targets": attr.string_list(
            doc = "Additional LLVM backends to configure alongside default backends.",
        ),
        "optional_targets": attr.string_list(
            doc = "LLVM backends whose libraries exist but are linked only by " +
                  "targets that depend on them explicitly. They stay out of " +
                  "the default aggregates, the LLVM shared library, " +
                  "Targets.def and the LLVM_HAS_<T>_TARGET defines.",
        ),
    },
)

# NOTE: exported as `llvm_configure` (not `llvm_project`) on purpose: the
# canonical bzlmod repo name is derived from the extension symbol, so this keeps
# it `@@+llvm_configure+llvm-project`. Renaming
# the symbol would rename the repo and break those references.
llvm_configure = module_extension(
    implementation = _llvm_project_impl,
    tag_classes = {"configure": _configure},
    doc = "Configures LLVM as `@llvm-project` with the selected backends.",
)
