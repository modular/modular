"""This rule hacks around a private API limitation in bazel by re-using the name of a library that is allowed to access the private API.

https://github.com/bazelbuild/bazel/pull/23838
"""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def link_hack(**kwargs):
    return cc_common.link(**kwargs)

def create_compile_variables_hack(**kwargs):
    return cc_common.create_compile_variables(**kwargs)

def strip_opts_hack(ctx):
    return ctx.fragments.cpp.strip_opts()
