load("@bazel_skylib//rules/directory:directory.bzl", "directory")

package(default_visibility = ["//visibility:public"])

exports_files(
    glob([
        "lib/clang/*/lib/*/libclang_rt.*",
        "bin/*",
    ]),
)

filegroup(
    name = "clang",
    srcs = glob(["bin/clang*"]),
)

filegroup(
    name = "ld",
    srcs = glob(["bin/*ld*"]),
)

filegroup(
    name = "include",
    srcs = [
        "lib/clang/20/include",
        "lib/clang/20/share",  # sanitizer default ignore lists
    ],
)

directory(
    name = "include_dir",
    srcs = [":include"],
)

filegroup(
    name = "resource_directory_filegroup",
    srcs = ["lib/clang/20"],
)

directory(
    name = "resource_directory",
    srcs = [":resource_directory_filegroup"],
)

filegroup(
    name = "bin",
    srcs = glob(["bin/**"]),
)

filegroup(
    name = "ar",
    srcs = ["bin/llvm-ar"],
)

filegroup(
    name = "as",
    srcs = ["bin/llvm-as"],
)

filegroup(
    name = "nm",
    srcs = ["bin/llvm-nm"],
)

filegroup(
    name = "objcopy",
    srcs = ["bin/llvm-objcopy"],
)

filegroup(
    name = "objdump",
    srcs = ["bin/llvm-objdump"],
)

filegroup(
    name = "profdata",
    srcs = ["bin/llvm-profdata"],
)

filegroup(
    name = "dwp",
    srcs = ["bin/llvm-dwp"],
)

filegroup(
    name = "ranlib",
    srcs = [
        "bin/llvm-ar",
        "bin/llvm-ranlib",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "bin/llvm-objcopy",
        "bin/llvm-strip",
    ],
)

filegroup(
    name = "clang-tidy",
    srcs = ["bin/clang-tidy"],
)
