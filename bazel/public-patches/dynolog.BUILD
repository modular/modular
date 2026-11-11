# BUILD for the @dynolog http_archive.
#
# dynolog is pulled ONLY for its header-only IPC fabric (Endpoint.h, Utils.h,
# FabricManager.h), which libkineto's IpcFabricConfigClient / DaemonConfigLoader
# include when built with ENABLE_IPC_FABRIC. We don't use libkineto's vendored
# copy (dynolog d2ffe0a, under libkineto/third_party/dynolog_headers at our
# pin): it is older than this pin, which is the one validated against Dynolog
# 0.6.0, and pytorch main has moved it out of the libkineto subtree.
# None of dynolog's daemon / CLI / Rust code is built here.
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "ipcfabric_headers",
    hdrs = glob(["dynolog/src/ipcfabric/*.h"]),
    # Two include roots: the repo root resolves the self-referential
    # "dynolog/src/ipcfabric/..." includes inside FabricManager.h; the leaf dir
    # resolves libkineto's plain #include "FabricManager.h".
    includes = [
        ".",
        "dynolog/src/ipcfabric",
    ],
    target_compatible_with = ["@platforms//os:linux"],
    # Only libkineto consumes these headers; cross-repo visibility works
    # the same way libkineto.BUILD already references @fmt.
    visibility = ["@libkineto//:__pkg__"],
)
