"""Wrapper macro for py_library"""

load("@rules_python//python:defs.bzl", "py_library")
load("//bazel/pip/pydeps:pydeps_test.bzl", "pydeps_test")
load(":modular_py_test.bzl", "modular_py_test")
load(":py_imports.bzl", "compute_py_imports")

def modular_py_library(
        name,
        visibility = None,
        ignore_extra_deps = [],
        ignore_unresolved_imports = [],
        imports = [],
        test_docstring_examples = False,
        docstring_example_deps = [],
        tags = [],
        **kwargs):
    """Creates a py_library target

    Args:
        name: The name of the underlying py_library
        visibility: The visibility of the target, defaults to public
        ignore_extra_deps: Forwarded to pydeps_test
        ignore_unresolved_imports: Forwarded to pydeps_test
        imports: The imports path. For max/python/max packages, this is
            automatically computed and should not be passed.
        test_docstring_examples: If True, generate a companion
            <name>.docstring_examples pytest target that runs Sybil on every
            docstring code-block example in this library's sources.
        docstring_example_deps: Extra runtime deps the examples import but the
            library itself cannot depend on (e.g. engine, which depends on
            graph). Ignored unless test_docstring_examples is True.
        tags: Tags to add to the target
        **kwargs: Extra arguments passed through to py_library
    """
    imports = compute_py_imports(native.package_name(), imports)

    if "manual" in tags:
        fail("modular_py_library targets cannot be manual. Remove 'manual' from the tags list.")

    py_library(
        name = name,
        visibility = visibility,
        imports = imports,
        tags = tags,
        **kwargs
    )

    if "no-pydeps" not in tags:
        pydeps_test(
            name = name + ".pydeps_test",
            deps = kwargs.get("deps", []),
            data = kwargs.get("data", []),
            ignore_extra_deps = ignore_extra_deps,
            ignore_unresolved_imports = ignore_unresolved_imports,
            target_compatible_with = select({
                # No point in running these, causes "error replanting symlinks" failures
                "//:asan": ["@platforms//:incompatible"],
                "//:ubsan": ["@platforms//:incompatible"],
                "//conditions:default": [],
            }),
            imports = imports if imports != None else [],
            srcs = kwargs.get("srcs", []) + kwargs.get("pyi_srcs", []),
            tags = ["pydeps"],
        )

    if test_docstring_examples:
        # Scope collection to this target's own sources: pass them explicitly
        # as pytest paths rather than the whole max/python/max tree, so the
        # test only runs examples from the opted-in target (not from its
        # dependencies). main + empty srcs load the collector as a plugin.
        example_srcs = [
            "{}/{}".format(native.package_name(), src)
            for src in kwargs.get("srcs", [])
            if src.endswith(".py")
        ]
        if not example_srcs:
            fail("test_docstring_examples = True requires Python 'srcs' to scan.")
        modular_py_test(
            name = name + ".docstring_examples",
            timeout = "long",
            srcs = [],
            main = "pytest_runner.py",
            args = example_srcs + [
                "-p",
                "sybil_collect",
                "--import-mode=importlib",
                "-o",
                "consider_namespace_packages=true",
            ],
            deps = [
                ":" + name,
                "//max/tests/docstring_examples:sybil_collect",
            ] + docstring_example_deps,
            tags = ["no-pydeps"],
        )
