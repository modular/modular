"""A macro for creating shared libraries based on our common patterns."""

load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_shared_library_info.bzl", "CcSharedLibraryInfo")
load(":modular_cc_library.bzl", "modular_cc_library")
load(":modular_strip_binary.bzl", "modular_strip_binary")

def _import_impl(ctx):
    shared = ctx.attr.shared[CcSharedLibraryInfo]
    return [
        CcInfo(
            compilation_context = ctx.attr.static[CcInfo].compilation_context,
            # The library and every shared library it links against, so a
            # binary that links this one can load them.
            linking_context = cc_common.create_linking_context(
                linker_inputs = depset(
                    [shared.linker_input] +
                    [dep.linker_input for dep in shared.dynamic_deps.to_list()],
                ),
            ),
        ),
        DefaultInfo(
            files = ctx.attr.shared[DefaultInfo].files,
            runfiles = ctx.attr.shared[DefaultInfo].default_runfiles,
        ),
    ]

# A shared library's headers, with the library and those it links against as
# prebuilt .so files.
_import = rule(
    implementation = _import_impl,
    attrs = {
        "shared": attr.label(mandatory = True, providers = [CcSharedLibraryInfo]),
        "static": attr.label(mandatory = True, providers = [CcInfo]),
    },
    provides = [CcInfo],
)

def modular_shared_library(
        name,
        deps = [],
        internal_deps = [],
        dynamic_deps = [],
        exports = [],
        exports_filter = [],
        includes = ["include"],
        output_name = None,
        shared_lib_name = None,
        linkopts = [],
        additional_linker_inputs = [],
        local_defines = [],
        with_import = False,
        visibility = None,
        tags = [],
        testonly = False,
        target_compatible_with = None,
        **kwargs):
    """Create a shared library target

    Creates these targets:
      - `name`: the cc_shared_library. Binaries and other shared libraries
        that link it list it in `dynamic_deps`.
      - `name.static`: the cc_library it is built from. Dependents that link
        `name` list this in `deps` for its headers, and rules_cc then links it
        through `name`. Without `name` in `dynamic_deps`, it is linked
        statically, as tests usually want.
      - `name.stripped`: the stripped library, for release packaging.
      - `name.import`: with `with_import`, the library and its headers as a
        prebuilt .so, linked like a system library: rules_cc doesn't check what
        it contains. For rules that only take CcInfo, such as Mojo's, and for
        leaf libraries that stand in for system ones.

    See oss/modular/bazel/docs/dynamic-linking.md.

    Args:
        name: The name of the shared library target
        deps: Deps of `name.static`
        internal_deps: Same as `deps`, but excluded for external builds.
        dynamic_deps: Shared libraries this one links against, such as
            @llvm-project//llvm:LLVM. Nothing is added implicitly.
        exports: Libraries this one owns besides `name.static`, which it links
            whole and exports. Its dependents that reach one of them link it
            from here instead of their own copy, as rules_cc requires of
            anything linked into this library. Their symbols must be visible.
        exports_filter: Patterns for more libraries to export, for those
            `exports` can't name, such as a dependency's private targets. They
            aren't linked whole, so they must be pulled in by what is.
        includes: Include search paths to propagate to dependents
        output_name: The name of the library on disk, without the `lib` prefix
            or extension. Defaults to `name`.
        shared_lib_name: The full file name of the library, overriding
            `output_name`.
        linkopts: Link flags for the shared library itself, not its dependents
        additional_linker_inputs: Files referenced by `linkopts`
        local_defines: Passed through to cc_library
        with_import: Whether to create `name.import`
        visibility: Set on all underlying targets
        tags: Set on all underlying targets
        testonly: Set on all underlying targets
        target_compatible_with: Set on all underlying targets
        **kwargs: Passed through to modular_cc_library, such as srcs and hdrs
    """
    common = dict(
        testonly = testonly,
        target_compatible_with = target_compatible_with,
        visibility = visibility,
    )

    modular_cc_library(
        name = name + ".static",
        deps = deps,
        internal_deps = internal_deps,
        includes = includes,
        local_defines = local_defines + ["MODULAR_BUILDING_LIBRARY"],
        tags = tags,
        **(common | kwargs)
    )

    cc_shared_library(
        name = name,
        deps = [":" + name + ".static"] + exports,
        dynamic_deps = dynamic_deps,
        exports_filter = exports_filter,
        shared_lib_name = shared_lib_name or "lib" + (output_name or name) + select({
            "@platforms//os:macos": ".dylib",
            "//conditions:default": ".so",
        }),
        user_link_flags = linkopts,
        additional_linker_inputs = additional_linker_inputs,
        tags = tags,
        **common
    )

    modular_strip_binary(
        name = name + ".stripped",
        binary = ":" + name,
        dsym_deps = [":" + name + ".static"] + exports,
        tags = tags + ["manual"],
        **common
    )

    if with_import:
        _import(
            name = name + ".import",
            shared = ":" + name,
            static = ":" + name + ".static",
            tags = tags,
            **common
        )
