load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "nanobind",
    srcs = glob(
        include = [
            "src/*.cpp",
            "src/*.h",
        ],
        exclude = [
            # This #includes all of the other cpp files
            "src/nb_combined.cpp",
        ],
    ),
    hdrs = glob(["include/**"]),
    additional_linker_inputs = select({
        "@platforms//os:macos": [":cmake/darwin-ld-cpython.sym"],
        "//conditions:default": [],
    }),
    copts = [
        "-fexceptions",
        "-frtti",
    ],
    includes = ["include"],
    linkopts = select({
        "@platforms//os:macos": [
            "-Wl,@$(location :cmake/darwin-ld-cpython.sym)",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@robin_map",
        "@rules_python//python/cc:current_py_cc_headers",
    ],
)

exports_files(["src/stubgen.py"])
