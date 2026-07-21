"""Stage a base Mojo source tree with overlay files shadowing it by path.

Produces staged `.mojo` files (not a compiled package) so a `mojo_library` can
consume them directly as `srcs`. All staged files share one staging root, so the
Mojo compiler resolves module paths relative to that root. This lets a build
flag swap individual stdlib files (for example `gpu/host/device_context.mojo`)
without disturbing the checked-in source tree.

`base_srcs` are staged at their path minus `base_strip_prefix` (when unset, the
parent directory of this target's package, so a filegroup in `.../std` stages
under `std/...` in both the internal and open-source repo layouts). Each entry
of `overlay_files` maps an overlay source file to the relative path it should
occupy in the staged tree; an overlay shadows any base file staged at the same
path. This allows an overlay file to live anywhere (it need not mirror the
target layout) and to be renamed onto the file it replaces.

Staged files are symlinks to their sources; nothing is copied or rewritten, so
every file compiles exactly as checked in. An overlay source whose imports
only resolve at its home location gets a staged-tree variant shadowing it by
path instead (see `_hal/_machine.mojo`).
"""

def _strip(path, prefix):
    if prefix and path.startswith(prefix):
        return path[len(prefix):]
    return path

def _single_file(target):
    files = target[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail("overlay_files target %s must provide exactly one file, got %d" % (
            target.label,
            len(files),
        ))
    return files[0]

def _mojo_overlay_srcs_impl(ctx):
    strip_prefix = ctx.attr.base_strip_prefix
    if not strip_prefix:
        package_parent, _, _ = ctx.label.package.rpartition("/")
        strip_prefix = package_parent + "/" if package_parent else ""

    # Map relative path -> source File. Base first, then overlay so it shadows.
    exclude = {p: True for p in ctx.attr.base_exclude}
    staged = {}
    for f in ctx.files.base_srcs:
        rel = _strip(f.short_path, strip_prefix)
        if rel in exclude:
            continue
        staged[rel] = f

    for target, rel in ctx.attr.overlay_files.items():
        staged[rel] = _single_file(target)

    # Stage in sorted-path order so the package root `__init__.mojo` (which sorts
    # first) is the consuming library's `srcs[0]`, fixing the compiler's root.
    outs = []
    for rel in sorted(staged.keys()):
        out = ctx.actions.declare_file(ctx.label.name + "/" + rel)
        ctx.actions.symlink(output = out, target_file = staged[rel])
        outs.append(out)

    return [DefaultInfo(files = depset(outs))]

mojo_overlay_srcs = rule(
    implementation = _mojo_overlay_srcs_impl,
    doc = "Stage base Mojo srcs with overlay files shadowing them by path.",
    attrs = {
        "base_srcs": attr.label_list(
            allow_files = [".mojo"],
            doc = "Base source files (for example the stdlib `**/*.mojo`).",
        ),
        "base_strip_prefix": attr.string(
            doc = "Prefix stripped from `base_srcs` short paths to get rel " +
                  "paths. Defaults to the parent directory of this target's " +
                  "package, which is correct for a base filegroup defined in " +
                  "the package being staged.",
        ),
        "base_exclude": attr.string_list(
            doc = "Relative paths (after `base_strip_prefix`) to drop from the " +
                  "staged tree entirely, for example backend modules that are " +
                  "orphaned once the overlay replaces their only importer.",
        ),
        "overlay_files": attr.label_keyed_string_dict(
            allow_files = [".mojo"],
            doc = "Maps an overlay source file to the staged relative path it " +
                  "should occupy, shadowing any base file at that path.",
        ),
    },
)
