"""Function to create tools definitions per platform."""

load("@rules_cc//cc/toolchains:tool.bzl", "cc_tool")
load("@rules_cc//cc/toolchains:tool_map.bzl", "cc_tool_map")
load("//bazel:config.bzl", "TOP_LEVEL_TAG")

PLATFORMS = [
    "linux-aarch64",
    "linux-x86_64",
    "macos",
]

# buildifier: disable=unnamed-macro
def declare_tools():
    for platform in PLATFORMS:
        _declare_tools(platform)

def _declare_tools(platform):
    cc_tool_map(
        name = "{}_tools".format(platform),
        tags = ["manual"],
        visibility = ["//bazel/internal/cc-toolchain:__subpackages__"],
        tools = {
            "@rules_cc//cc/toolchains/actions:ar_actions": ":{}-llvm-ar".format(platform),
            "@rules_cc//cc/toolchains/actions:assembly_actions": ":clang",
            "@rules_cc//cc/toolchains/actions:c_compile": ":clang",
            "@rules_cc//cc/toolchains/actions:cpp_compile_actions": ":clang++",
            "@rules_cc//cc/toolchains/actions:link_actions": ":linker_driver",
            "@rules_cc//cc/toolchains/actions:objc_compile": ":clang",
            "@rules_cc//cc/toolchains/actions:objcopy_embed_data": ":{}-llvm-objcopy".format(platform),
            "@rules_cc//cc/toolchains/actions:strip": ":{}-llvm-strip".format(platform),
            "@rules_cc//cc/toolchains/actions:dwp": ":{}-llvm-dwp".format(platform),
        },
    )

    cc_tool(
        name = "{}-llvm-ar".format(platform),
        src = "@clang-{}//:bin/llvm-ar".format(platform),
        tags = ["manual"],
    )

    cc_tool(
        name = "{}-llvm-objcopy".format(platform),
        src = "@clang-{}//:bin/llvm-objcopy".format(platform),
        tags = ["manual"],
    )

    cc_tool(
        name = "{}-llvm-strip".format(platform),
        src = "@clang-{}//:bin/llvm-strip".format(platform),
        tags = ["manual"],
    )

    cc_tool(
        name = "{}-llvm-dwp".format(platform),
        src = "@clang-{}//:bin/llvm-dwp".format(platform),
        tags = ["manual"],
    )

    cc_tool(
        name = "{}-clang-tidy".format(platform),
        src = "@clang-{}//:bin/clang-tidy".format(platform),
        tags = [
            "manual",
            TOP_LEVEL_TAG,  # Used in .bazelrc
        ],
    )

    native.alias(
        name = "{}-builtin_headers".format(platform),
        actual = "@clang-{}//:include".format(platform),
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )

    native.alias(
        name = "{}-resource_directory".format(platform),
        actual = "@clang-{}//:resource_directory".format(platform),
        tags = ["manual"],
        visibility = ["//visibility:private"],
    )

    cc_tool(
        name = "{}-clang-format".format(platform),
        src = "@clang-{}//:bin/clang-format".format(platform),
        tags = ["manual"],
    )

    cc_tool(
        name = "{}-clangd".format(platform),
        src = "@clang-{}//:bin/clangd".format(platform),
        tags = ["manual"],
    )

    cc_tool(
        name = "{}-llvm-install-name-tool".format(platform),
        src = "@clang-{}//:bin/llvm-install-name-tool".format(platform),
        data = ["@clang-{}//:bin/llvm-objcopy".format(platform)],
        tags = ["manual"],
    )

    cc_tool(
        name = "{}-llvm-otool".format(platform),
        src = "@clang-{}//:bin/llvm-otool".format(platform),
        data = ["@clang-{}//:bin/llvm-objdump".format(platform)],
        tags = ["manual"],
    )
