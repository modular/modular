"""Module extension that fetches the LLVM source."""

# BEGIN_GENERATED
# NOTE: Use 'update-llvm' to update these values
LLVM_COMMIT = "fa8c67382e4fa68010fce98bbd267f91b397f80e"

LLVM_SHA = "b4ab9e2d62be5eb4c0d90bc0cf4bffce8819560f77a7fa480073bd40fbb0f4ad"
# END_GENERATED

PATCHES = [
    "//bazel/public-patches:llvm-lldb-exports.patch",
    # https://github.com/llvm/llvm-project/pull/153352
    # https://linear.app/modularml/issue/MOCO-2322/llvm-upstream-change-conflicting-with-internal-code-that-addresses
    "//bazel/public-patches:llvm-machinefunction-sti-ref-to-ptr.patch",
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
    # The overlay's lldb:HostMacOSXObjCXX lists the objcxx headers in
    # non_arc_srcs, but Bazel's objc_library rejects header files there
    # ("source file ... is misplaced here") because that attribute is only
    # for .m/.mm sources compiled without ARC. The headers are just inputs
    # for the .mm files, so list them in srcs instead.
    "//bazel/public-patches:llvm-lldb-macosx-objcxx-headers.patch",
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
    # Soft-promoted half (bf16/f16) SCALAR_TO_VECTOR operands hit the
    # "Do not know how to soft promote this operator's operand!" fatal error
    # in SoftPromoteHalfOperand, which has no case for SCALAR_TO_VECTOR. The
    # X86 backend forms v8bf16 = scalar_to_vector for soft-promoted bf16
    # selects (llvm/llvm-project bec73b4b5c2f), so host code that vectorizes
    # bf16 math crashes at DAG->DAG ISel. Promote the scalar operand to i16,
    # build the integer vector, and bitcast back, mirroring BUILD_VECTOR.
    "//bazel/public-patches:llvm-softpromote-scalar-to-vector.patch",
    # llvm/llvm-project 3fdac7de5496 ("[X86] Expand f32-to-bf16 conversions
    # without libcalls") keeps v8bf16 legal on every AVX target, but the only
    # v8bf16/v16bf16 X86ISD::VBROADCAST selection patterns are guarded by
    # [HasBWI, HasVLX] in X86InstrAVX512.td. On AVX2-only targets (e.g.
    # x86-64-v3) combineBroadcasts' broadcast(bitcast(src)) ->
    # bitcast(broadcast(src)) rewrite turns a v8i16 broadcast of a bitcast
    # v8bf16 operand into a v8bf16-typed X86ISD::VBROADCAST, and ISel aborts
    # with "Cannot select: v8bf16 = X86ISD::VBROADCAST". Mirror the v8f16/
    # v16f16 patterns: select v8bf16/v16bf16 broadcasts to VPBROADCASTW(Y)
    # under [HasAVX2, NoVLX_Or_NoBWI].
    "//bazel/public-patches:llvm-avx2-bf16-broadcast.patch",
    # Provides targets to enable use of `libLLVM.so` and related changes.
    # TODO: Upstream
    "//bazel/public-patches:llvm-shared.patch",
]

# Upstream lit test inputs that no target reads: about three quarters of the
# archive's files. Extracting them dominated fetching LLVM. The Bazel overlay
# builds libraries from libc/test, lld/test and mlir/test, so those stay.
_EXCLUDED_DIRS = [
    "clang-tools-extra/test",
    "clang/test",
    "compiler-rt/test",
    "libcxx/test",
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
