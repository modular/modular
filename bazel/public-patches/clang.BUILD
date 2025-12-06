load("@bazel_skylib//rules/directory:directory.bzl", "directory")

package(default_visibility = ["//visibility:public"])

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
    srcs = glob([
        "lib/clang/*/include/**",
        "lib/clang/*/share/**/*.txt",  # sanitizer default ignore lists
    ]),
)

directory(
    name = "include_dir".format(platform),
    srcs = [":include"],
)

directory(
    name = "resource_directory",
    srcs = glob(["lib/clang/*/**"]),
)

filegroup(
    name = "bin",
    srcs = glob(["bin/**"]),
)

filegroup(
    name = "lib",
    srcs = glob(
        [
            "lib/clang/*/lib/**/*.a",  # sanitizers
            "lib/clang/*/lib/**/*.o",  # crtbegin.o
            "lib/clang/*/lib/**/*.so",  # sanitizers linux
            "lib/clang/*/lib/**/*.dylib",  # sanitizers macOS
            "lib/clang/*/lib/**/*.syms",  # sanitizers syms files used during linking
        ],
        allow_empty = True,
    ),
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
