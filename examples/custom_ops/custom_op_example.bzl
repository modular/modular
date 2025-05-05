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
            # Ensure that the `mojo` tool is available.
            "//KGEN/tools/mojo",
        ] + extra_data,
        env = {
            # Note: This relative path works because the default working directory
            #   for a `modular_py_binary` target is the runfiles root directory.
            "PATH": "KGEN/tools/mojo:/usr/bin:/bin",
        },
        use_sitecustomize = True,
        imports = ["."],
        mojo_deps = [
            "@mojo//:layout",
            "//SDK/lib/API/mojo/max/compiler",
            "//SDK/lib/API/mojo/max/max",
            "//SDK/lib/API/mojo/max/tensor",
            "@mojo//:stdlib",
        ],
        deps = [
            "//SDK/lib/API/python/max/driver",
            "//SDK/lib/API/python/max/engine",
            "//SDK/lib/API/python/max/graph",
            requirement("numpy"),
        ] + extra_deps,
    )

    # Run each example as a simple non-zero-exit-code test.
    if create_test:
        modular_run_binary_test(
            name = name + ".example-test",
            args = [],
            binary = name,
        )
