"""Module extension that fetches the LLVM source."""

# BEGIN_GENERATED
# NOTE: Use 'update-llvm' to update these values
LLVM_COMMIT = "6941494e2ae550d753728ba13454311b344d82cf"

LLVM_SHA = "fd9814a9c57f7340cd3b4b2c60777214eb2d193458e434afc131de2c96a61a2c"
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
    # llvm/llvm-project#225158 gives the llvm_libxml2_enabled config_setting
    # group visibility to clang, lld/test and lldb, but leaves its match_any
    # members private. Bazel checks the members' visibility from the consuming
    # target, so lldb:ConfigHeader fails to see llvm_libxml2_auto. Let the
    # three members inherit the package's public default.
    "//bazel/public-patches:llvm-libxml2-config-setting-visibility.patch",
    # Default the overlay's feature flags to what every build wants, so no rc
    # flag names an @llvm-project label. Parsing such a flag fetches the LLVM
    # source before analysis can start.
    "//bazel/public-patches:llvm-flag-defaults.patch",
]

# Upstream lit test inputs that no target reads: about three quarters of the
# archive's files. Extracting them dominated fetching LLVM. The Bazel overlay
# builds libraries from libc/test, lld/test and mlir/test, so those stay.
_EXCLUDED_DIRS = [
    "clang-tools-extra/test",
    "clang/test",
    "compiler-rt/test",
    "libcxx/test",
    "lldb/test",
    "llvm/test",
]

def _llvm_archive_impl(rctx):
    archive = "llvm.tar.gz"
    rctx.download(
        url = rctx.attr.url,
        output = archive,
        sha256 = rctx.attr.sha256,
    )
    rctx.report_progress("Extracting LLVM")
    args = ["tar", "-xzf", archive, "--strip-components=1"]
    for d in rctx.attr.excluded_dirs:
        args.append("--exclude={}/{}".format(rctx.attr.strip_prefix, d))
    result = rctx.execute(args)
    if result.return_code:
        fail("Failed to extract LLVM:\n{}{}".format(result.stdout, result.stderr))
    rctx.delete(archive)

    for patch in rctx.attr.patches:
        rctx.patch(patch, strip = 1)

    rctx.file("BUILD.bazel", "exports_files(glob([\"**\"]))")
    return rctx.repo_metadata(reproducible = True)

_llvm_archive = repository_rule(
    implementation = _llvm_archive_impl,
    doc = """Like http_archive, but skips directories while extracting.

Uses the host's tar, since http_archive has no way to exclude paths.""",
    attrs = {
        "excluded_dirs": attr.string_list(),
        "patches": attr.label_list(),
        "sha256": attr.string(mandatory = True),
        "strip_prefix": attr.string(mandatory = True),
        "url": attr.string(mandatory = True),
    },
)

def _llvm_source_impl(module_ctx):
    patches = list(PATCHES)
    for mod in module_ctx.modules:
        for tag in mod.tags.configure:
            patches.extend([str(p) for p in tag.extra_patches])

    _llvm_archive(
        name = "llvm-raw",
        excluded_dirs = _EXCLUDED_DIRS,
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
