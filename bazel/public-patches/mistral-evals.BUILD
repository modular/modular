load("@@//bazel:api.bzl", "modular_py_binary", "requirement")

modular_py_binary(
    name = "evaluate",
    srcs = glob(["eval/**/*.py"]),
    imports = ["."],
    main = "eval/run.py",
    visibility = ["//visibility:public"],
    deps = [
        requirement("datasets"),
        requirement("fire"),
        requirement("numpy"),
        requirement("openai"),
        requirement("pillow"),
        requirement("tqdm"),
    ],
)
