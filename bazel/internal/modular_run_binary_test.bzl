"""A test rule that runs a given modular_py_binary target."""

load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "env_for_available_tools", "get_default_exec_properties", "get_default_test_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility
load(":binary_test.bzl", "binary_test")

def modular_run_binary_test(
        name,
        binary,
        data = [],
        env = {},
        external_noop = False,  # buildifier: disable=unused-variable
        target_compatible_with = [],
        gpu_constraints = [],
        exec_properties = {},
        tags = [],
        toolchains = [],
        **kwargs):
    """Run a binary as a test, the test result will be the exit code of the binary.

    Args:
        name: Name of the test target
        binary: Label of the binary to run
        data: Runtime data required by the binary
        env: Environment variables to set
        external_noop: Ignored, for compatibility with the external repo
        target_compatible_with: See upstream docs
        gpu_constraints: GPU requirements for the tests
        exec_properties: https://bazel.build/reference/be/common-definitions#common-attributes
        tags: See upstream docs
        toolchains: Extra toolchains
        **kwargs: Passed through to the test target
    """

    validate_gpu_tags(tags, gpu_constraints)

    binary_test(
        name = name,
        binary = binary,
        data = data + [
            "//bazel/internal:asan-suppressions.txt",
            "//bazel/internal:lsan-suppressions.txt",
        ],
        env = GPU_TEST_ENV | get_default_test_env(exec_properties) | env_for_available_tools() | env,
        tags = tags,
        target_compatible_with = target_compatible_with + gpu_constraints,
        exec_properties = get_default_exec_properties(tags, gpu_constraints) | exec_properties,
        toolchains = toolchains + [
            "//bazel/internal:current_gpu_toolchain",
            "//bazel/internal:lib_toolchain",
        ],
        **kwargs
    )
