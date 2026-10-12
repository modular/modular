load("@//bazel:api.bzl", "modular_py_library", "requirement")

modular_py_library(
    name = "sphinx-markdown-builder",
    srcs = glob(["sphinx_markdown_builder/**/*.py"]),
    imports = ["sphinx_markdown_builder"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("docutils"),
        requirement("tabulate"),
        requirement("sphinx"),
    ],
)
