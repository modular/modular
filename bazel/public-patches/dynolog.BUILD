# BUILD for the @dynolog http_archive.
#
# dynolog is pulled ONLY for its header-only IPC fabric (Endpoint.h, Utils.h,
# FabricManager.h), which libkineto's IpcFabricConfigClient / DaemonConfigLoader
# include when built with ENABLE_IPC_FABRIC. libkineto's own source tarball omits
# these because upstream keeps dynolog as a git submodule, and source archives do
# not include submodules. None of dynolog's daemon / CLI / Rust code is built here.
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
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    # Only libkineto consumes these headers; cross-repo visibility works
    # the same way libkineto.BUILD already references @fmt.
    visibility = ["@libkineto//:__pkg__"],
)
