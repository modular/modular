"""Wrapper for mojo_binary to add internal logic."""

load("@rules_mojo//mojo:mojo_binary.bzl", _upstream_mojo_binary = "mojo_binary")

def mojo_binary(
        name,
        tags = [],
        **kwargs):
    _upstream_mojo_binary(
        name = name,
        tags = ["mojo-fixits"] + tags,
        **kwargs
    )
