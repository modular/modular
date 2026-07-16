"""Wrapper around upstream mojo_library rule to add documentation generation."""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_mojo//mojo:mojo_library.bzl", _upstream_mojo_library = "mojo_library")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load(":cc_transition.bzl", "asan_to_production_config_select", "cc_transition")
load(":mojo_doc.bzl", "mojo_doc")

# A self-transitioning variant of the upstream mojo_library that, under
# `--config=asan`, builds with the production (non-instrumented) compiler via
# cc_transition -- fast, while producing the same config-invariant `.mojopkg`.
# See cc_transition.bzl for the asan-only gating (`new_config`).
_transitioned_mojo_library = rule(
    implementation = lambda ctx: ctx.super(),
    parent = _upstream_mojo_library,
    attrs = {
        "new_config": attr.string(mandatory = True),
    },
    cfg = cc_transition,
)

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
        use_production_compiler_for_asan = False,
        copts = [],
        tags = []):
    """
    Precompiles sources into a mojoc file.

    Args:
        name: Forwarded to `mojo_library`
        srcs: Forwarded to Forwarded to all subtargets
        deps: Forwarded to Forwarded to all subtargets
        data: Forwarded to `mojo_library`
        validate_missing_docs: Forwarded to `mojo_docs`
        docs_base_path: Forwarded to `mojo_docs`
        docs_title: Forwarded to `mojo_docs`
        docs_hosted_on_mojolang: Forwarded to `mojo_docs`
        show_stability_markers: Forwarded to `mojo_docs`
        testonly: Forwarded to `mojo_library`
        visibility: Forwarded to all subtargets
        additional_compiler_inputs: Forwarded to `mojo_library`
        use_production_compiler_for_asan:
            Use a production build of Mojo to build this target when in asan mode.
            This is a speed optimization if the coverage is not worth the slowdown.
        copts: Forwarded to `mojo_library`
        tags: Forwarded to all subtargets
    """
    library_rule = _transitioned_mojo_library if use_production_compiler_for_asan else _upstream_mojo_library
    library_kwargs = {"new_config": asan_to_production_config_select} if use_production_compiler_for_asan else {}
    library_rule(
        name = name,
        srcs = srcs,
        data = data,
        deps = deps,
        visibility = visibility,
        testonly = testonly,
        tags = ["mojo-fixits"] + tags,
        additional_compiler_inputs = additional_compiler_inputs,
        copts = copts,
        **library_kwargs
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
