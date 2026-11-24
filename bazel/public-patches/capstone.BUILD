# Based on https://github.com/google/silifuzz/blob/e43fa71ad2a550d8dd0afa775284c85d898617bf/third_party/BUILD.capstone
# Should roughly match https://github.com/wolfpld/tracy/blob/16bac5f80790fb29ca2bd1eab903f02e8ea0732a/cmake/vendor.cmake#L31-L53

load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "capstone",
    srcs = [
        "MCInst.c",
        "MCInstrDesc.c",
        "MCRegisterInfo.c",
        "SStream.c",
        "cs.c",
        "utils.c",
    ] + glob([
        "arch/*/*.c",
        "**/*.h",
        "arch/*/*.inc",
    ]),
    hdrs = glob(["include/capstone/*.h"]),
    copts = [
        "-Wno-ignored-qualifiers",
        "-Wno-missing-field-initializers",
        "-DCAPSTONE_X86_ATT_DISABLE",
    ],
    includes = [
        "include",
        "include/capstone",
    ],
    visibility = ["@tracy//:__subpackages__"],
)
