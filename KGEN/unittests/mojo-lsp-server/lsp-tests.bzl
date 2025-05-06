"""Helper functions for creating Mojo LSP Server tests targets."""

load("//bazel:api.bzl", "modular_python_binding_library_test")

def lsp_test(name, pattern):
    modular_python_binding_library_test(
        name = name,
        size = "large",
        srcs = native.glob(
            [pattern],
        ),
        data = native.glob([
            "inputs/*.mojo",
            "inputs_with_package/*.mojo",
        ]) + [
            "//KGEN/tools/mojo-lsp-server",
            "//open-source/max/mojo/stdlib/stdlib:stdlib_srcs",
            "@crashpad//:modular-crashpad-handler",
        ],
        env = {
            "MODULAR_MOJO_MAX_IMPORT_PATH": "open-source/max/mojo/stdlib",
            "MODULAR_PATH": ".",
            "PRESERVE_LSP_IO_FILES": "1",
        },
        mojo_deps = [
            "@mojo//:stdlib",
            "//SDK/lib/API/mojo/max/tensor",
            "//open-source/max/mojo/kernels/src/extensibility/compiler_internal",
        ],
        py_deps = [],
        tags = [
            "no-sandbox",  # The LSP server currently has issues with symlinks and non-canonical paths
        ],
        deps = [
            ":Support",
            "//Support:Globals",
        ],
    )
