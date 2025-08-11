"""Custom Op example helpers to reduce boilerplate in BUILD.bazel file."""

load("//bazel:api.bzl", "modular_py_binary", "modular_run_binary_test", "requirement")

def custom_op_example_py_binary(
        name,
        srcs,
        extra_data = [],
        extra_deps = [],
        target_compatible_with = None,
        tags = []):
    modular_py_binary(
        name = name,
        srcs = srcs,
        data = [
            ":kernel_sources",
        ] + extra_data,
        imports = ["."],
        mojo_deps = [
            "@mojo//:compiler",
            "@mojo//:layout",
            "@mojo//:stdlib",
            "@mojo//:tensor_internal",
        ],
        deps = [
            "//SDK/lib/API/python/max/driver",
            "//SDK/lib/API/python/max/engine",
            "//SDK/lib/API/python/max/graph",
            "//open-source/max/mojo/python/mojo",
            requirement("numpy"),
        ] + extra_deps,
        target_compatible_with = target_compatible_with,
    )

    # Run each example as a simple non-zero-exit-code test.
    modular_run_binary_test(
        name = name + ".example-test",
        args = [],
        binary = name,
        tags = ["gpu"] + tags,
    )
