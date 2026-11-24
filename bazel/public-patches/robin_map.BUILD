load("@rules_cc//cc:defs.bzl", "cc_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "robin_map",
    hdrs = [
        "include/tsl/robin_growth_policy.h",
        "include/tsl/robin_hash.h",
        "include/tsl/robin_map.h",
        "include/tsl/robin_set.h",
    ],
    includes = ["."],
    strip_include_prefix = "include",
)
