"""Wrapper for the upstream cc_test rule to apply modular specific defaults."""

load("@cc_compatibility_proxy//:proxy.bzl", _upstream_cc_test = "cc_test")
load("@rules_mojo//mojo/private:transitions.bzl", "python_version_transition")  # buildifier: disable=bzl-visibility
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "RUNTIME_SANITIZER_DATA", "env_for_available_tools", "get_default_exec_properties", "get_default_test_env", "runtime_sanitizer_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility
load(":modular_cc_binary.bzl", "modular_cc_binary")
load(":mojo_test_environment.bzl", "mojo_test_environment")

cc_test = rule(
    implementation = lambda ctx: ctx.super(),
    parent = _upstream_cc_test,
    attrs = {
        "python_version": attr.string(),
    },
    cfg = python_version_transition,
)

def modular_cc_test(
        name,
        copts = [],
        deps = [],
        data = [],
        mojo_deps = [],
        env = {},
        exec_properties = {},
        tags = [],
        target_compatible_with = [],
        gpu_constraints = [],
        toolchains = [],
        size = None,
        timeout = None,
        python_version = None,
        **kwargs):
    """A wrapper for cc_test

    https://bazel.build/reference/be/c-cpp#cc_test

    Args:
        name: The test target's name
        copts: See cc_test docs
        deps: See cc_test docs
        data: See cc_test docs
        mojo_deps: mojo_library targets the test depends on at runtime
        env: See cc_test docs
        exec_properties: Remote exec resources https://www.buildbuddy.io/docs/rbe-platforms/#runner-resource-allocation
        tags: Tags to set on the underlying targets
        target_compatible_with: https://bazel.build/extending/platforms#skipping-incompatible-targets
        gpu_constraints: GPU requirements for the tests
        toolchains: See cc_test docs
        size: See cc_test docs
        timeout: See cc_test docs
        python_version: The python version to use for all transitive deps of the test.
        **kwargs: Everything else passed through to modular_cc_binary with the exception of `size` and `timeout`
    """
    if not deps:
        fail("A test with no deps does nothing")

    validate_gpu_tags(tags, target_compatible_with + gpu_constraints)

    extra_toolchains = ["//bazel/internal:current_gpu_toolchain"]
    extra_data = RUNTIME_SANITIZER_DATA + [
        "@llvm-project//llvm:llvm-symbolizer",
    ]
    extra_env = env_for_available_tools() | GPU_TEST_ENV | runtime_sanitizer_env() | get_default_test_env(exec_properties) | {
        "GTEST_COLOR": "1",
        "CC_CODE_COVERAGE_SCRIPT": "bazel/internal/cc-toolchain/collect-cc-coverage.sh",
    }
    if mojo_deps:
        env_name = name + ".mojo_test_env"
        extra_toolchains.append(env_name)
        extra_data += [env_name]  # buildifier: disable=list-append
        extra_env |= {"MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)"}
        mojo_test_environment(
            name = env_name,
            data = mojo_deps,
            testonly = True,
        )

    kwargs = kwargs | {
        "copts": copts + [
            "-Wno-global-constructors",
        ],
        "testonly": True,
        "data": data + extra_data,
        "env": extra_env | env,
        "deps": deps + [
            "@llvm-project//third-party/unittest:gmock",
            "@llvm-project//third-party/unittest:gtest",
            "@llvm-project//third-party/unittest:gtest_main",
        ],
        "target_compatible_with": target_compatible_with + gpu_constraints,
    }

    modular_cc_binary(
        name = name + ".debug",
        tags = tags + [
            "manual",
            ALLOW_UNUSED_TAG,
        ],
        toolchains = toolchains + extra_toolchains,
        **kwargs
    )

    cc_test(
        name = name,
        exec_properties = get_default_exec_properties(tags, gpu_constraints) | exec_properties,
        tags = tags,
        timeout = timeout,
        toolchains = toolchains + extra_toolchains,
        size = size,
        python_version = python_version,
        **kwargs
    )
