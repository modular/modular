load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "ppqsort",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["@tracy//:__subpackages__"],
)
