"""Wrapper for clang_tidy_test to handle our requirements."""

load("@bazel_clang_tidy//clang_tidy:clang_tidy_test.bzl", "clang_tidy_test")

_clang_tidy_test = rule(
    parent = clang_tidy_test,
    implementation = lambda ctx: ctx.super(),
    exec_groups = {
        "test": exec_group(
            # NOTE: this excludes the default test toolchain so that this test
            # runs on the first execution platform, and therefore can run
            # remotely regardless of the local platform.
            toolchains = [
                "@bazel_tools//tools/cpp:toolchain_type",
            ],
        ),
    },
)

def modular_clang_tidy_test(
        name,
        hdrs,
        srcs,
        copts,
        tags,
        additional_compiler_inputs = []):
    """Create a clang_tidy_test target

    Args:
        name: The name prefix of the test target
        hdrs: Header files to lint
        srcs: Source files to lint
        copts: Compiler flags required to build this library
        tags: Tags that can potentially disable clang-tidy
        additional_compiler_inputs: Files that might be required to fully type check the srcs
    """

    # Skip targets if no files would be linted
    if "no-clang-tidy" in tags:
        return
    if "no-clang-tidy-headers" in tags and not srcs:
        return

    # Libraries that only combine deps or textual_hdrs don't get clang-tidy support
    if not srcs and not hdrs:
        return

    clang_tidy_tags = []
    if "manual" in tags:
        clang_tidy_tags.append("manual")

    tests = []
    for src in srcs + hdrs:
        # Ignore generated files
        if ":" in src:
            continue

        # Ignore files that clang_tidy_test would ignore anyways
        if not src.endswith((".h", ".cpp", ".c")):
            continue

        if "no-clang-tidy-headers" in tags and src.endswith((".h")):
            continue

        test_name = "{}_{}.tidy".format(name, src)
        tests.append(test_name)
        _clang_tidy_test(
            name = test_name,
            deps = [name],
            data = srcs + hdrs + additional_compiler_inputs,
            srcs = [src],
            copts = copts,
            tags = clang_tidy_tags + [
                "clang-tidy",
                "lint-test",
            ],
            clang_tidy_config = "@//:clang_tidy_config",
            clang_tidy_executable = "@//bazel/internal/cc-toolchain/tools:clang-tidy",
            clang_tidy_resource_dir = "@//bazel/internal/cc-toolchain/tools:builtin_headers",
            visibility = ["//visibility:private"],
        )

    if tests:
        native.test_suite(
            name = name + ".tidy",
            tests = tests,
            tags = ["manual"],
            visibility = ["//visibility:private"],
        )
