"""Create a modular_py_test for every supported python version."""

load("@module_versions//:config.bzl", "DEFAULT_PYTHON_VERSION", "PYTHON_VERSIONS_DOTTED")
load(":config.bzl", "python_version_name", "python_version_tags")
load(":modular_py_test.bzl", "modular_py_test")

def modular_multi_py_version_test(name, tags = [], **kwargs):
    """Creates a test that runs against multiple Python versions.

    Args:
        name: The name to use as the prefix of the target names
        tags: Tags to set on the underlying test targets
        **kwargs: Other arguments to pass through to the underlying test targets
    """

    targets = []
    for python_version in PYTHON_VERSIONS_DOTTED:
        # Delete redundant pydeps tests
        pydeps_tag = [] if python_version == DEFAULT_PYTHON_VERSION else ["no-pydeps"]
        target_name = python_version_name(name, python_version)
        targets.append(target_name)
        modular_py_test(
            name = target_name,
            python_version = python_version,
            tags = tags + pydeps_tag + python_version_tags(python_version),
            **kwargs
        )

    native.test_suite(
        name = name + ".suite",
        tags = ["manual"],
        tests = targets,
    )
