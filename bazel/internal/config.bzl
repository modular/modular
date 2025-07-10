"""Private bazel configuration used internally by rules and macros."""

def env_for_available_tools(
        *,
        location_specifier = "rootpath",  # buildifier: disable=unused-variable
        os = "unknown"):  # buildifier: disable=unused-variable
    return {}
