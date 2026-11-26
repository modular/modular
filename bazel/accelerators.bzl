"""Constraint values of the platforms with an accelerator, kept out of config.bzl so that an edit rebuilds only what loads them."""

ACCELERATOR_CONSTRAINTS = ["//:has_gpu"]
