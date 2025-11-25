"""Create a local repository for the macOS SDK."""

# Based on https://cs.opensource.google/pigweed/pigweed/+/main:pw_toolchain/xcode.bzl

def _macos_sysroot_repository_impl(rctx):
    if rctx.os.name != "mac os x":
        # Write an empty file that will fail if used by otherwise allows analysis
        rctx.file("BUILD.bazel", """\
load("@bazel_skylib//rules/directory:directory.bzl", "directory")

directory(
    name = "root",
    srcs = [],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sysroot-macos",
    srcs = [],
    visibility = ["//visibility:public"],
)
""")
        return

    developer_dir = rctx.getenv("DEVELOPER_DIR")
    result = rctx.execute(
        ["/usr/bin/xcrun", "--show-sdk-path", "--sdk", "macosx"],
        environment = {"DEVELOPER_DIR": developer_dir},
    )
    if result.return_code != 0:
        fail("Failed locating macOS SDK: {}".format(result.stderr))

    sdk_path = rctx.path(result.stdout.strip())
    for child in sdk_path.readdir(watch = "no"):
        rctx.symlink(child, child.basename)
    rctx.file("BUILD.bazel", """\
load("@bazel_skylib//rules/directory:directory.bzl", "directory")

# NOTE: Ruby.framework has infinite symlink loops, so we have to avoid that in our globs
# NOTE: Including the smallest set of files is ideal to avoid more cache usage
_INCLUDES = [
    "usr/lib/**",
    "usr/include/**",
    "System/Library/Frameworks/CoreFoundation.framework/**",
    "System/Library/Frameworks/Foundation.framework/**",
    "System/Library/Frameworks/Metal.framework/**",
]

directory(
    name = "root",
    srcs = glob(_INCLUDES),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "sysroot-macos",
    srcs = glob(_INCLUDES),
    visibility = ["//visibility:public"],
)
""")

macos_sysroot_repository = repository_rule(
    implementation = _macos_sysroot_repository_impl,
    environ = [
        "XCODE_VERSION",
        "DEVELOPER_DIR",
    ],
    local = True,
    configure = True,
)
