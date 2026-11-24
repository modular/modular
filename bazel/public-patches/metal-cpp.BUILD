load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "metal-cpp",
    hdrs = glob([
        "Foundation/*.hpp",
        "Metal/*.hpp",
        "QuartzCore/*.hpp",
    ]),
    includes = [
        ".",
        "Foundation",
        "Metal",
        "QuartzCore",
    ],
    linkopts = [
        "-Wl,-framework,CoreGraphics",
        "-Wl,-framework,Metal",
    ],
    target_compatible_with = select({
        # Metal-cpp is only compatible with macOS
        "@platforms//os:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)
