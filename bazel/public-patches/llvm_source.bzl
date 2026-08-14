"""Module extension that fetches the LLVM source."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# BEGIN_GENERATED
# NOTE: Use 'update-llvm' to update these values
LLVM_COMMIT = "ec26997e2e4606d97918a4a082c4f93ca38a6f46"

LLVM_SHA = "7636ff70c60a2933a91362932127478b7b24a610ef1f01afa58fc2a4ecb125ed"
# END_GENERATED

PATCHES = [
    "//bazel/public-patches:llvm-lldb-exports.patch",
    # https://github.com/llvm/llvm-project/pull/153352
    # https://linear.app/modularml/issue/MOCO-2322/llvm-upstream-change-conflicting-with-internal-code-that-addresses
    "//bazel/public-patches:llvm-machinefunction-sti-ref-to-ptr.patch",
    # https://github.com/llvm/llvm-project/pull/175650
    "//bazel/public-patches:llvm-fix-lldb-dap-console.patch",
    # Fix heap corruption in ObjectFileELF::GetModuleSpecifications: use a
    # local DataExtractor copy instead of mutating the shared extractor_sp,
    # which invalidated other DataExtractors sharing the same buffer and
    # caused glibc malloc to detect a corrupted double-linked list on teardown.
    # https://github.com/llvm/llvm-project/pull/188978
    # https://github.com/llvm/llvm-project/issues/190255
    "//bazel/public-patches:llvm-elf-extractor-local-copy.patch",
    # Revert the config.bzl musl/gnu select() from llvm/llvm-project#207295,
    # which keys HAVE_BACKTRACE/BACKTRACE_HEADER/HAVE_MALLINFO on
    # @llvm//platforms/config:{musl,gnu}. That package is not present in the
    # repo produced by our llvm_configure module extension, so the select()
    # fails to resolve ("No repository visible as '@llvm'"). Restore the
    # unconditional glibc/macOS defines, which are correct for our builds
    # (we do not target musl).
    "//bazel/public-patches:llvm-config-musl-select.patch",
    # OrcTargetProcess's OrcRTBootstrap.cpp now includes
    # RTBridge/SPS/Calls.h (llvm/llvm-project#213265), which pulls in Core.h
    # and, transitively, most of the top-level Orc/*.h headers plus
    # JITLink/JITLinkDylib.h. TargetProcess depends on :OrcShared, not
    # :OrcJIT/:JITLink (the targets that already glob those headers) --
    # and can't depend on either without a dependency cycle, since both
    # :OrcJIT and :JITLink already depend on :OrcTargetProcess (:JITLink
    # also depends on :OrcShared directly). Mirror their header globs onto
    # :OrcShared's hdrs (a plain file list, not a link dependency) so the
    # headers are visible where they're actually included from; this is
    # header-only exposure, the .cpp only uses header-defined type aliases,
    # not anything requiring OrcJIT's/JITLink's .cpp-defined symbols.
    "//bazel/public-patches:llvm-orcshared-sps-rtbridge-headers.patch",
]

def _llvm_source_impl(module_ctx):
    patches = list(PATCHES)
    for mod in module_ctx.modules:
        for tag in mod.tags.configure:
            patches.extend([str(p) for p in tag.extra_patches])

    http_archive(
        name = "llvm-raw",
        build_file_content = "exports_files(glob([\"**\"]))",
        patch_strip = 1,
        patches = patches,
        sha256 = LLVM_SHA,
        strip_prefix = "llvm-project-{}".format(LLVM_COMMIT),
        url = "https://github.com/llvm/llvm-project/archive/{}.tar.gz".format(LLVM_COMMIT),
    )

    return module_ctx.extension_metadata(reproducible = True)

_configure = tag_class(
    attrs = {
        "extra_patches": attr.label_list(
            doc = "Additional LLVM patches to apply on top of default patches.",
        ),
    },
)

llvm_source = module_extension(
    implementation = _llvm_source_impl,
    tag_classes = {"configure": _configure},
    doc = "Fetches the patched LLVM source archive as `@llvm-raw`.",
)
