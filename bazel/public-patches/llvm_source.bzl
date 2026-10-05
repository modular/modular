"""Module extension that fetches the LLVM source."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# BEGIN_GENERATED
# NOTE: Use 'update-llvm' to update these values
LLVM_COMMIT = "a83a3a4ebd64653dbe6a85819996b3bcfd6753ec"

LLVM_SHA = "c26cddd8981c07cd562d05ef2baa58c2b30e0dd3537b8c42971463680d54dc1f"
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
    # Revert llvm/llvm-project#218490, which changed ELFSectionHeaderInfo's
    # section_name from ConstString to std::string. ConstString is a pointer
    # into a never-freed intern pool; std::string owns its buffer, so the
    # header info now frees memory and reintroduces the same corrupted
    # double-linked list on teardown that the patch above fixes. Regenerated
    # over llvm/llvm-project#224170 and llvm/llvm-project#224407, which moved
    # Section names to std::string as well; keeping only the header info on
    # ConstString is still enough to avoid the crash.
    # TODO(MOTO-1590): drop once the ownership is sound across DSO boundaries.
    "//bazel/public-patches:llvm-revert-lldb-elf-section-name-string.patch",
    # Drop the overlay's @rules_cc//cc/libc:{musl,glibc} select()s: the
    # config.bzl HAVE_BACKTRACE/BACKTRACE_HEADER/HAVE_MALLINFO defines
    # (llvm/llvm-project#207295, #223549) and the Support musl stack-size
    # linkopt (llvm/llvm-project#223989). Those labels need rules_cc >= 0.2.25;
    # we pin 0.2.18, so they do not resolve. The unconditional glibc/macOS
    # defines are correct for our builds (we do not target musl). Drop this
    # once rules_cc is bumped to 0.2.25 or newer.
    "//bazel/public-patches:llvm-config-musl-select.patch",
    # llvm/llvm-project#225158 gives the llvm_libxml2_enabled config_setting
    # group visibility to clang, lld/test and lldb, but leaves its match_any
    # members private. Bazel checks the members' visibility from the consuming
    # target, so lldb:ConfigHeader fails to see llvm_libxml2_auto. Let the
    # three members inherit the package's public default.
    "//bazel/public-patches:llvm-libxml2-config-setting-visibility.patch",
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
