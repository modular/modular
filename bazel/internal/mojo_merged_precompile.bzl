"""Build a mojoc by staging an ordered stack of overlay layers and compiling."""

load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_mojo//mojo:providers.bzl", "MojoInfo")
load(":overlay_layer.bzl", "OverlayLayerInfo")

def _format_include(f):
    return "-I" + f.path

def _mojo_merged_precompile_impl(ctx):
    mojo_toolchain = ctx.toolchains["@rules_mojo//:toolchain_type"].mojo_toolchain_info

    # Use 'std.mojoc' as the filename to ensure the package name is 'std'
    mojoc = ctx.actions.declare_file("std.mojoc")

    # Merge the ordered layers last-writer-wins per staged path. `stage_root`
    # roots the tree at the package name (empty for the bare stdlib root this
    # rule compiles from).
    exclude = {p: True for p in ctx.attr.exclude}
    root = ctx.attr.stage_root
    staged = {}
    for layer in ctx.attr.layers:
        for rel, f in layer[OverlayLayerInfo].entries.items():
            staged[rel] = f

    # Stage the merged tree under a private root, but at the same repo-relative
    # subpath the primary stdlib compiles from. Compiling from
    # `<staging>/Mojo/stdlib/std` and stripping only `<staging>/`
    # reproduces the primary stdlib's `Mojo/stdlib/std/...` locs,
    # so those instantiations unify with symbols baked into oss-std-built
    # kernel packages (@@//max:linalg) this overlay std is linked against. A
    # private staging root keeps this off the real base sources bazel
    # materializes at `Mojo/stdlib/std` in the execroot.
    staging_root = ctx.label.name + ".staging"
    stdlib_subpath = "Mojo/stdlib/std"
    staging_std = staging_root + "/" + stdlib_subpath

    mojodeps = []
    import_paths = []
    cc_infos = []
    for dep in ctx.attr.deps + mojo_toolchain.implicit_deps:
        if MojoInfo in dep:
            info = dep[MojoInfo]
            mojodeps.append(info.mojodeps)
            import_paths.append(info.import_paths)
            cc_infos.append(info.ccdeps)
        if CcInfo in dep:
            cc_infos.append(dep[CcInfo])
    transitive_mojodeps = depset(transitive = mojodeps)

    # Action: Merge and Compile
    args = ctx.actions.args()
    args.add(mojo_toolchain.mojo)
    args.add(mojoc)
    args.add(staging_std)

    # Pack the rest of the arguments
    args.add_all(mojo_toolchain.precompile_copts)
    args.add_all(ctx.attr.copts)
    args.add_all(transitive_mojodeps, map_each = _format_include)
    for bc in ctx.files.bitcode_libs:
        args.add("--bitcode-libs", bc)

    script_lines = [
        "#!/bin/bash",
        "set -e",
        'MOJO_COMPILER="$1"; shift',
        'OUTPUT_MOJOC="$1"; shift',
        'STAGING_STD="$1"; shift',
        'REST_ARGS=("$@")',
        'mkdir -p "$STAGING_STD"',
    ]

    # Copy each staged file in sorted-path order (deterministic action). A later
    # layer already won the merge above, so every path appears once here.
    layer_inputs = []
    for rel in sorted(staged.keys()):
        if rel in exclude:
            continue
        f = staged[rel]
        layer_inputs.append(f)
        dest = root + rel
        script_lines.append('mkdir -p "$STAGING_STD/$(dirname %s)"' % dest)
        script_lines.append('cp "%s" "$STAGING_STD/%s"' % (f.path, dest))

    script_lines.extend([
        # `-strip-file-prefix` is canonicalized to an ABSOLUTE path and only
        # strips filenames the compiler recorded as absolute. The recorded
        # path follows the package argument's form, so pass the package as an
        # absolute path (forcing absolute filenames) and strip the absolute
        # staging root; a relative argument records relative filenames the
        # absolute prefix can't match, silently leaving `std.staging/` on the
        # loc. Both leave the baked loc at `Mojo/stdlib/std/...`.
        'STAGING_ABS="$(pwd)/$STAGING_STD"',
        'STRIP_PREFIX="$(pwd)/${STAGING_STD%%/*}/"',
        # We use PATH=/dev/null to ensure the compiler doesn't find host stdlib.
        'PATH=/dev/null "$MOJO_COMPILER" precompile "-strip-file-prefix=$STRIP_PREFIX" -o "$OUTPUT_MOJOC" "${REST_ARGS[@]}" "$STAGING_ABS"',
    ])

    script = ctx.actions.declare_file(ctx.label.name + "_build.sh")
    ctx.actions.write(script, "\n".join(script_lines), is_executable = True)

    all_inputs = layer_inputs + ctx.files.bitcode_libs + [script, mojo_toolchain.mojo]
    ctx.actions.run(
        executable = script,
        inputs = depset(all_inputs, transitive = [transitive_mojodeps]),
        tools = mojo_toolchain.all_tools,
        outputs = [mojoc],
        arguments = [args],
        mnemonic = "MojoMergedPrecompile",
        progress_message = "%{label} precompiling merged mojo sources",
        env = {
            "MODULAR_CRASH_REPORTING_ENABLED": "false",
            "PATH": "/usr/bin:/bin",
            "TEST_TMPDIR": ".",
        },
        use_default_shell_env = True,
        toolchain = "@rules_mojo//:toolchain_type",
        execution_requirements = {
            "supports-path-mapping": "1",
        },
    )

    return [
        DefaultInfo(files = depset([mojoc])),
        MojoInfo(
            import_paths = depset([struct(package = mojoc, import_path = ".")]),
            mojodeps = depset([mojoc]),
            ccdeps = cc_common.merge_cc_infos(cc_infos = cc_infos),
        ),
    ]

mojo_merged_precompile = rule(
    implementation = _mojo_merged_precompile_impl,
    attrs = {
        "layers": attr.label_list(
            providers = [OverlayLayerInfo],
            doc = "Ordered overlay layers merged into the staged tree; a later " +
                  "layer shadows an earlier one where their staged paths collide.",
        ),
        "stage_root": attr.string(
            doc = "Prepended to every staged path. Empty (the default) stages " +
                  "at the bare stdlib root this rule compiles from.",
        ),
        "exclude": attr.string_list(
            doc = "Staged relative paths to drop from the tree, for example " +
                  "backend modules orphaned once an overlay replaces their " +
                  "only importer.",
        ),
        "bitcode_libs": attr.label_list(
            allow_files = [".bc"],
            doc = "Bitcode library files to pass to mojo precompile.",
        ),
        "copts": attr.string_list(
            default = [],
            doc = "Extra flags forwarded to `mojo precompile` (e.g. `--ignore-deprecated=...`).",
        ),
        "deps": attr.label_list(
            providers = [[MojoInfo], [CcInfo]],
            doc = "Mirrors `mojo_library.deps`. Dependencies providing CcInfo " +
                  "are not used to precompile this target, they are propagated " +
                  "to the compile and link actions of whatever depends on it.",
        ),
    },
    toolchains = ["@rules_mojo//:toolchain_type"],
)
