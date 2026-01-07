load("@//bazel:api.bzl", "modular_py_library", "requirement")

modular_py_library(
    name = "breathe",
    srcs = glob(["breathe/**/*.py"]),
    imports = ["./breathe"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("docutils"),
        requirement("jinja2"),
        requirement("markupsafe"),
        requirement("pygments"),
        requirement("sphinx"),
    ],
)
