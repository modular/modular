"""Wrapper for mojo_shared_library to add internal logic."""

load("@rules_mojo//mojo:mojo_shared_library.bzl", _upstream_mojo_shared_library = "mojo_shared_library")

def mojo_shared_library(
        name,
        **kwargs):
    _upstream_mojo_shared_library(
        name = name,
        **kwargs
    )
