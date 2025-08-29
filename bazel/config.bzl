"""Shared config that needs to be referenced by BUILD.bazel files but isn't important enough to be in api.bzl"""

# Used for linting unused targets, top level targets are potentially used
# externally, and therefore their deps are all considered used
TOP_LEVEL_TAG = "top-level"

# Used for linting unused targets, these targets might be unused and that's
# allowed, used sparingly, primarily used for macros that expand to multiple
# targets, some of which are optional
ALLOW_UNUSED_TAG = "maybe-unused"

# Default GPU memory for scheduling remote exec tests
DEFAULT_GPU_MEMORY = "0.8"
