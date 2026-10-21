"""Build jemalloc with a page size that suits the target CPU."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

_LG_PAGE = "@jemalloc//settings/platform:lg_page"

def _lg_page_transition_impl(_settings, attr):
    return {_LG_PAGE: attr.lg_page}

# jemalloc's own transition only defaults this by OS, and it leaves a value we
# set here alone, so this is where the CPU gets a say.
_lg_page_transition = transition(
    implementation = _lg_page_transition_impl,
    inputs = [],
    outputs = [_LG_PAGE],
)

def _jemalloc_with_lg_page_impl(ctx):
    target = ctx.attr.src[0]
    providers = [DefaultInfo, CcInfo, InstrumentedFilesInfo, OutputGroupInfo]
    return [target[provider] for provider in providers if provider in target]

jemalloc_with_lg_page = rule(
    doc = """Forwards a jemalloc library built for the given base 2 log page size.

    jemalloc bakes its page size in at compile time and refuses to start on a
    host whose pages are larger than that, so an aarch64 build has to assume
    the 64 KiB maximum. A build for 64 KiB pages still runs on smaller pages,
    which is why one binary can cover both. Pass "__auto__" to keep jemalloc's
    own default.

    Supported hosts follow from that: x86_64 with 4 KiB pages, the only base
    page size Linux offers there, and aarch64 with 4, 16 or 64 KiB pages, every
    size the Linux arm64 kernel can be configured for. The cost of the aarch64
    choice is that on 4 KiB hosts jemalloc manages memory in 64 KiB units, so
    its page accounting is coarser and peak RSS can run somewhat higher than a
    native 4 KiB build would.
    """,
    implementation = _jemalloc_with_lg_page_impl,
    attrs = {
        "lg_page": attr.string(mandatory = True),
        "src": attr.label(
            cfg = _lg_page_transition,
            mandatory = True,
            providers = [CcInfo],
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
