"""Wrapper for mojo_binary to add internal logic."""

load("@rules_mojo//mojo:mojo_binary.bzl", _upstream_mojo_binary = "mojo_binary")

def mojo_binary(
        name,
        linkopts = [],
        **kwargs):
    _upstream_mojo_binary(
        name = name,
        linkopts = linkopts + select({
            "@platforms//os:linux": [
                "-Wl,-rpath,$ORIGIN/../lib",
            ],
            "@platforms//os:macos": [
                "-Wl,-rpath,@loader_path/../lib",
            ],
        }),
        **kwargs
    )
