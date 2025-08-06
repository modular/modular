"""Private bazel configuration used internally by rules and macros."""

load("@with_cfg.bzl//with_cfg/private:select.bzl", "decompose_select_elements")  # buildifier: disable=bzl-visibility
load("//bazel:config.bzl", "DEFAULT_GPU_MEMORY")  # buildifier: disable=bzl-visibility

GPU_TEST_ENV = {
    "ASAN_OPTIONS": "$(GPU_ASAN_OPTIONS),suppressions=$(execpath //bazel/internal:asan-suppressions.txt)",
    "GPU_ENV_DO_NOT_USE": "$(GPU_CACHE_ENV)",
    "LSAN_OPTIONS": "suppressions=$(execpath //bazel/internal:lsan-suppressions.txt)",
}

def _get_all_constraints(constraints):
    """Extract all possible constraints from the target's 'target_compatible_with'.

    This is complicated because if the 'target_compatible_with' is a select,
    you cannot check if it has a value. This uses an upstream hack to parse the
    select and return all possible values, even if they are not in effect.
    """
    flattened_constraints = []
    for in_select, elements in decompose_select_elements(constraints):
        if type(elements) == type([]):
            flattened_constraints.extend(elements)
        else:
            if in_select and (elements == {} or elements == {"//conditions:default": []}):
                fail("Empty select, delete it")
            flattened_constraints.extend(elements.keys())
            for selected_constraints in elements.values():
                flattened_constraints.extend(selected_constraints)

    return flattened_constraints

def validate_gpu_tags(tags, target_compatible_with):
    """Fail if configured gpu_constraints + tags aren't supported.

    Args:
        tags: The target's 'tags'
        target_compatible_with: The target's 'target_compatible_with'
    """
    has_tag = "gpu" in tags
    if has_tag:
        return

    has_gpu_constraints = any([
        constraint.endswith(("_gpu", "_gpus"))
        for constraint in _get_all_constraints(target_compatible_with)
    ])
    if has_gpu_constraints:
        fail("tests that have 'gpu_constraints' must specify 'tags = [\"gpu\"],' to be run on CI")

def get_default_exec_properties(tags, target_compatible_with):
    """Return exec_properties that should be shared between different test target types.

    Args:
        tags: The target's 'tags'
        target_compatible_with: The target's 'target_compatible_with'

    Returns:
        A dictionary that should be added to exec_properties of the test target
    """
    gpu_constraints = _get_all_constraints(target_compatible_with)

    exec_properties = {}
    if "requires-network" in tags:
        exec_properties["test.dockerNetwork"] = "bridge"

    if "@//:has_multi_gpu" in gpu_constraints or "//:has_multi_gpu" in gpu_constraints:
        exec_properties["test.resources:gpu-1"] = "0"
        exec_properties["test.resources:gpu-2"] = "0.01"

    if "@//:has_4_gpus" in gpu_constraints or "//:has_4_gpus" in gpu_constraints:
        exec_properties["test.resources:gpu-1"] = "0"
        exec_properties["test.resources:gpu-2"] = "0.01"
        exec_properties["test.resources:gpu-4"] = "0.01"

    return exec_properties

def get_default_test_env(exec_properties):
    """Get environment variables that should be shared between different test target types.

    Args:
        exec_properties: The target's 'exec_properties'

    Returns:
        A dictionary that should be added to the test target's 'env'
    """

    # TODO(MOTO-1278): 0.6 accounts for unknown overhead
    gpu_memory_limit = float(exec_properties.get("test.resources:gpu-memory", DEFAULT_GPU_MEMORY))
    adjusted_gpu_memory_limit = gpu_memory_limit - 0.6
    if adjusted_gpu_memory_limit < 0.0:
        fail("GPU memory limit must be at least 1 GiB, got: {}".format(gpu_memory_limit))

    return select({
        "@//:has_gpu": {
            "MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_ONLY": "true",
            "MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE": "{}".format(int(adjusted_gpu_memory_limit * 1073741824.0)),
            "MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_CHUNK_PERCENT": "100",
        },
        "//conditions:default": {},
    })

def env_for_available_tools(
        *,
        location_specifier = "rootpath",  # buildifier: disable=unused-variable
        os = "unknown"):  # buildifier: disable=unused-variable
    return {}
