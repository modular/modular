package(default_visibility = ["//visibility:public"])

# The wasi-libc wasm64 sysroot is built by a single action
# (//Mojo/lib/CompilerRT/wasi:wasi_sysroot) that runs the upstream Makefile
# on a copied source tree, so the whole checkout is exported as one
# filegroup. `Makefile` is also exported individually so the action can
# locate the source root via its dirname.
exports_files(["Makefile"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
