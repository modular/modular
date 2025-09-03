"""A test rule that executes a mojo_binary, passing its output to FileCheck."""

load("@rules_mojo//mojo:mojo_binary.bzl", "mojo_binary")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "get_default_exec_properties", "get_default_test_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility

def mojo_filecheck_test(
        name,
        srcs,
        copts = [],
        data = [],
        deps = [],
        enable_assertions = True,
        env = {},
        expect_crash = False,
        expect_fail = False,
        main = None,
        size = None,
        tags = [],
        target_compatible_with = [],
        exec_properties = {},
        toolchains = [],
        **kwargs):
    """Creates a test that runs a mojo_binary and checks its output with FileCheck.

    Args:
        name: The name of the test.
        copts: Compiler options for the mojo_binary.
        data: Files needed by the test at runtime
        enable_assertions: Whether to enable assertions for the mojo_binary.
        env: Environment variables to set for the binary and test.
        expect_crash: Whether to expect the mojo_binary to crash.
        expect_fail: Whether to expect the mojo_binary to fail with `not`.
        srcs: The source files for the mojo_binary.
        size: The size of the test.
        main: The main source file for the mojo_binary. Only needed if multiple source files are passed.
        deps: Dependencies for the mojo_binary.
        tags: Tags to set on all underlying targets
        target_compatible_with: See upstream docs
        exec_properties: See upstream docs
        toolchains: See upstream docs
        **kwargs: Additional arguments to pass to the mojo_binary and test rules.
    """
    validate_gpu_tags(tags, target_compatible_with)

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
        data = data + [
            "//bazel/internal:lsan-suppressions.txt",
        ],
        env = env | GPU_TEST_ENV,
        testonly = True,
        enable_assertions = enable_assertions,
        tags = tags,
        target_compatible_with = target_compatible_with,
        toolchains = toolchains + ["//bazel/internal:current_gpu_toolchain"],
        **kwargs
    )

    if expect_crash and expect_fail:
        fail("Only one of 'expect_crash' or 'expect_fail' can be True.")

    default_exec_properties = get_default_exec_properties(tags, target_compatible_with)
    sh_test(
        name = name,
        srcs = ["//bazel/internal:mojo-filecheck-test"],
        size = size,
        data = data + srcs + [
            name + ".binary",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:not",
            "//bazel/internal:lsan-suppressions.txt",
        ],
        env = env | GPU_TEST_ENV | get_default_test_env(exec_properties) | {
            "BINARY": "$(location :{}.binary)".format(name),
            "EXPECT_CRASH": "1" if expect_crash else "0",
            "EXPECT_FAIL": "1" if expect_fail else "0",
            "FILECHECK": "$(location @llvm-project//llvm:FileCheck)",
            "NOT": "$(location @llvm-project//llvm:not)",
            "SOURCE": "$(location {})".format(filecheck_src),
        },
        tags = ["filecheck"] + tags,
        target_compatible_with = target_compatible_with,
        exec_properties = default_exec_properties | exec_properties,
        toolchains = toolchains + ["//bazel/internal:current_gpu_toolchain"],
        **kwargs
    )
