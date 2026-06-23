"""Wrapper around upstream mojo_library rule to add documentation generation."""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_mojo//mojo:mojo_library.bzl", _upstream_mojo_library = "mojo_library")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load(":mojo_doc.bzl", "mojo_doc")

def mojo_library(
        name,
        srcs,
        data = [],
        deps = [],
        validate_missing_docs = True,
        docs_base_path = "",
        docs_title = "",
        docs_hosted_on_mojolang = False,
        show_stability_markers = "none",
        testonly = False,
        visibility = None,
        additional_compiler_inputs = [],
        copts = [],
        tags = []):
    _upstream_mojo_library(
        name = name,
        srcs = srcs,
        data = data,
        deps = deps,
        visibility = visibility,
        testonly = testonly,
        tags = ["mojo-fixits"] + tags,
        additional_compiler_inputs = additional_compiler_inputs,
        copts = copts,
    )

    if not testonly:
        mojo_doc(
            name = name + ".docs",
            srcs = srcs,
            deps = deps,
            validate_missing_docs = validate_missing_docs,
            docs_base_path = docs_base_path,
            docs_title = docs_title,
            docs_hosted_on_mojolang = docs_hosted_on_mojolang,
            show_stability_markers = show_stability_markers,
            visibility = visibility,
            tags = [ALLOW_UNUSED_TAG] + tags,
            testonly = testonly,
        )

        build_test(
            name = name + ".docs_test",
            targets = [name + ".docs"],
            tags = ["mojo-docs", "lint-test"] + tags,
            visibility = ["//visibility:private"],
        )
