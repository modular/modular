"""Custom Op example helpers to reduce boilerplate in BUILD.bazel file."""

load("//bazel:api.bzl", "modular_py_binary", "modular_run_binary_test", "requirement")

def custom_op_example_py_binary(
        name,
        srcs,
        create_test = True,
        extra_data = [],
        extra_deps = []):
    modular_py_binary(
        name = name,
        srcs = srcs,
        data = [
            ":kernel_sources",
        ] + extra_data,
        imports = ["."],
        mojo_deps = [
            "//max:compiler",
            "//max:layout",
            "@mojo//:stdlib",
            "//max:tensor",
        ],
        deps = [
            "//max/python/max/driver",
            "//max/python/max/engine",
            "//max/python/max/graph",
            requirement("numpy"),
        ] + extra_deps,
        visibility = ["//visibility:private"],
        testonly = True,
    )

    # Run each example as a simple non-zero-exit-code test.
    if create_test:
        modular_run_binary_test(
            name = name + ".example-test",
            args = [],
            binary = name,
        )
