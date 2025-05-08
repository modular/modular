"""A test rule that executes a mojo_binary, passing its output to FileCheck."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_mojo//mojo:mojo_binary.bzl", "mojo_binary")

def mojo_filecheck_test(name, srcs, deps = [], enable_assertions = True, **kwargs):
    if len(srcs) != 1:
        fail("Only a single source file may be passed")

    mojo_binary(
        name = name + ".binary",
        srcs = srcs,
        deps = deps,
        testonly = True,
        enable_assertions = enable_assertions,
        **kwargs
    )

    native.sh_test(
        name = name,
        srcs = ["//bazel/internal:mojo-filecheck-test"],
        args = [paths.join(native.package_name(), src) for src in srcs],
        data = srcs + [
            name + ".binary",
            "@llvm-project//llvm:FileCheck",
        ],
        env = {
            "BINARY": "$(location :{}.binary)".format(name),
            "FILECHECK": "$(location @llvm-project//llvm:FileCheck)",
        },
        **kwargs
    )
