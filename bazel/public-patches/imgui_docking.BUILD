load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Upstream imgui's `docking` branch ships no Bazel build files, so we provide
# our own here. `imgui_docking` is consumed via the `@imgui_docking` repo name
# defined by the `http_archive` rule in //MODULE.bazel.

cc_library(
    name = "imgui_docking",
    srcs = [
        "imgui.cpp",
        "imgui_demo.cpp",
        "imgui_draw.cpp",
        "imgui_tables.cpp",
        "imgui_widgets.cpp",
        "misc/freetype/imgui_freetype.cpp",
    ],
    hdrs = [
        "imconfig.h",
        "imgui.h",
        "imgui_internal.h",
        "imstb_rectpack.h",
        "imstb_textedit.h",
        "imstb_truetype.h",
        "misc/freetype/imgui_freetype.h",
    ],
    defines = [
        "IMGUI_USE_WCHAR32",
        "IMGUI_ENABLE_FREETYPE",
    ],
    include_prefix = "imgui",
    includes = ["."],
    deps = ["@freetype"],
)

cc_library(
    name = "platform-glfw",
    srcs = ["backends/imgui_impl_glfw.cpp"],
    hdrs = ["backends/imgui_impl_glfw.h"],
    include_prefix = "imgui",
    includes = [
        ".",
        "backends",
    ],
    linkopts = select({
        "@platforms//os:linux": ["-lpthread"],
        "//conditions:default": [],
    }),
    target_compatible_with = select({
        "@platforms//os:osx": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":imgui_docking",
        "@glfw",
    ],
)

cc_library(
    name = "renderer-opengl3",
    srcs = ["backends/imgui_impl_opengl3.cpp"],
    hdrs = [
        "backends/imgui_impl_opengl3.h",
        "backends/imgui_impl_opengl3_loader.h",
    ],
    include_prefix = "imgui",
    includes = [
        ".",
        "backends",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "-lGL",
            "-ldl",
        ],
        "//conditions:default": [],
    }),
    deps = [":imgui_docking"],
)
