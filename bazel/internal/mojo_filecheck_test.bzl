"""A test rule that executes a mojo_binary, passing its output to FileCheck."""

load("@rules_mojo//mojo:mojo_binary.bzl", "mojo_binary")

def mojo_filecheck_test(name, srcs, copts = [], deps = [], enable_assertions = True, expect_crash = False, main = None, size = None, **kwargs):
    """Creates a test that runs a mojo_binary and checks its output with FileCheck.

    Args:
        name: The name of the test.
        copts: Compiler options for the mojo_binary.
        enable_assertions: Whether to enable assertions for the mojo_binary.
        expect_crash: Whether to expect the mojo_binary to crash.
        srcs: The source files for the mojo_binary.
        size: The size of the test.
        main: The main source file for the mojo_binary. Only needed if multiple source files are passed.
        deps: Dependencies for the mojo_binary.
        **kwargs: Additional arguments to pass to the mojo_binary and test rules.
    """
    filecheck_src = srcs[0]
    if len(srcs) > 1:
        if not main:
            fail("If multiple source files are passed, a main file must be specified.")
        filecheck_src = main

    mojo_binary(
        name = name + ".binary",
        copts = copts,
        srcs = srcs,
        main = main,
        deps = deps,
        testonly = True,
        enable_assertions = enable_assertions,
        **kwargs
    )

    native.sh_test(
        name = name,
        srcs = ["//bazel/internal:mojo-filecheck-test"],
        size = size,
        data = srcs + [
            name + ".binary",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:not",
        ],
        env = {
            "BINARY": "$(location :{}.binary)".format(name),
            "EXPECT_CRASH": "1" if expect_crash else "0",
            "FILECHECK": "$(location @llvm-project//llvm:FileCheck)",
            "NOT": "$(location @llvm-project//llvm:not)",
            "SOURCE": "$(location {})".format(filecheck_src),
        },
        **kwargs
    )
