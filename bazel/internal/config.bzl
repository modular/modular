"""Private bazel configuration used internally by rules and macros."""

GPU_TEST_ENV = {
    "ASAN_OPTIONS": "$(GPU_ASAN_OPTIONS),suppressions=$(execpath //bazel/internal:asan-suppressions.txt)",
    "GPU_ENV_DO_NOT_USE": "$(GPU_CACHE_ENV)",
    "LSAN_OPTIONS": "suppressions=$(execpath //bazel/internal:lsan-suppressions.txt)",
}

def validate_gpu_tags(tags, gpu_constraints):
    """Fail if configured gpu_constraints + tags aren't supported.

    Args:
        tags: The target's 'tags'
        gpu_constraints: The target's 'gpu_constraints'
    """
    has_tag = "gpu" in tags
    if not has_tag and gpu_constraints:
        fail("tests that have 'gpu_constraints' must specify 'tags = [\"gpu\"],' to be run on CI")

def get_default_exec_properties(tags, gpu_constraints):
    """Return exec_properties that should be shared between different test target types.

    Args:
        tags: The target's 'tags'
        gpu_constraints: The target's 'gpu_constraints'

    Returns:
        A dictionary that should be added to exec_properties of the test target
    """

    exec_properties = {}
    if "requires-network" in tags:
        exec_properties["test.dockerNetwork"] = "bridge"

    if "@//:has_multi_gpu" in gpu_constraints or "//:has_multi_gpu" in gpu_constraints:
        exec_properties["test.resources:gpu-2"] = "0.01"

    if "@//:has_4_gpus" in gpu_constraints or "//:has_4_gpus" in gpu_constraints:
        exec_properties["test.resources:gpu-4"] = "0.01"

    return exec_properties

def env_for_available_tools(
        *,
        location_specifier = "rootpath",  # buildifier: disable=unused-variable
        os = "unknown"):  # buildifier: disable=unused-variable
    return {}
