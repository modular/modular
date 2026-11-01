"""Per-file overlays for opted-in packages.

A package listed in `OVERLAY_PACKAGES` gets a `:overlay` label flag. Pointing it
at an `overlay_layer` replaces any of the package's source files, keyed by
package-relative path, without editing the package's BUILD file. A plugin
mirrors the package layout and declares the layer:

    overlay_layer(
        name = "overlay",
        srcs = glob(["include/**", "lib/**"]),
        strip_prefix = "plugins/<p>/GraphCompiler/",
        visibility = ["//GraphCompiler:__pkg__"],
    )

then builds with `--//GraphCompiler:overlay=//plugins/<p>/GraphCompiler:overlay`
(typically from a `.bazelrc` config). `//GraphCompiler/test/overlay` is a
working example. The package's `include/` and `lib/` files are exported to
`//plugins/...` so a replacement can be generated from its base file.

The macros in `//bazel:api.bzl` that take source paths (`modular_cc_library`,
`td_library`, `gentbl_cc_library`, `gentbl_filegroup`) route each path through
a per-file `<path>.overlay` target. It forwards the base file untouched unless
the layer replaces it, in which case the replacement is staged at
`overlay_tree/<path>`. Include directories get a matching `overlay_tree/<dir>`
entry placed ahead of the original, so `#include` and TableGen `include` resolve
the replacement first, also from targets that don't list the file.

Only replacing existing files is supported. A layer entry naming a path that
no wrapped macro routes (a typo, a removed file, or a file only reachable
through a native rule) fails analysis. Same-package `filegroup`s passed to a
wrapped macro are expanded so their files are routed too.
"""

load(":overlay_layer.bzl", "OverlayLayerInfo", "overlay_layer")

# Packages whose source files can be overlaid.
OVERLAY_PACKAGES = [
    "GraphCompiler",
]

_FLAG = "overlay"
_NONE = "overlay.none"
_CHECK = "overlay.check"
_TREE = "overlay_tree"

def _overlay_check_impl(ctx):
    routed = {path: True for path in ctx.attr.routed}
    unknown = sorted([
        rel
        for rel in ctx.attr.overlay[OverlayLayerInfo].entries.keys()
        if rel not in routed
    ])
    if unknown:
        fail(("overlay %s replaces paths that no overlay-aware target in //%s " +
              "consumes:\n  %s") % (
            ctx.attr.overlay.label,
            ctx.label.package,
            "\n  ".join(unknown),
        ))
    return []

_overlay_check = rule(
    implementation = _overlay_check_impl,
    doc = "Fails analysis if the overlay names a path no target routes.",
    attrs = {
        "routed": attr.string_list(),
        "overlay": attr.label(providers = [OverlayLayerInfo]),
    },
)

def _overlay_check_finalizer_impl(name, visibility, overlay):  # buildifier: disable=unused-variable
    # A finalizer runs after the rest of the package, so it sees every
    # per-file target the wrapped macros created.
    _overlay_check(
        name = name,
        routed = [
            rule["path"]
            for rule in native.existing_rules().values()
            if rule["name"].endswith(".overlay") and "path" in rule
        ],
        overlay = overlay,
        visibility = ["//visibility:public"],
    )

_overlay_check_finalizer = macro(
    implementation = _overlay_check_finalizer_impl,
    finalizer = True,
    attrs = {
        "overlay": attr.label(configurable = False, mandatory = True),
    },
)

def _overlay_file_impl(ctx):
    replacement = ctx.attr.overlay[OverlayLayerInfo].entries.get(ctx.attr.path)
    if not replacement:
        return [DefaultInfo(files = depset(ctx.files.src))]

    staged = ctx.actions.declare_file(_TREE + "/" + ctx.attr.path)
    ctx.actions.symlink(output = staged, target_file = replacement)
    return [DefaultInfo(files = depset([staged]))]

_overlay_file = rule(
    implementation = _overlay_file_impl,
    doc = "Forwards `src`, or stages the overlay's replacement for `path`.",
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "path": attr.string(mandatory = True),
        "overlay": attr.label(providers = [OverlayLayerInfo], mandatory = True),
        # Only depended on so that its analysis-time check runs.
        "check": attr.label(mandatory = True),
    },
)

def _overlay_filegroup_impl(ctx):
    return [DefaultInfo(files = depset(ctx.files.srcs))]

_overlay_filegroup = rule(
    implementation = _overlay_filegroup_impl,
    doc = "The per-file overlay targets of an expanded `filegroup`.",
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        # Unused at analysis; keeps the expanded filegroup a dependency, so it
        # isn't reported as unused.
        "base": attr.label(mandatory = True),
    },
)

def _enabled():
    return native.package_name() in OVERLAY_PACKAGES

def _ensure_package_targets():
    if native.existing_rule(_FLAG):
        return
    overlay_layer(name = _NONE)

    # Lets an overlay build a replacement from its base file, for example by
    # patching it, instead of carrying a copy that drifts.
    native.exports_files(
        native.glob(
            ["include/**", "lib/**"],
            # Docs aren't overlay inputs, and exporting them would count them
            # as referenced.
            exclude = ["**/README*", "**/*.md", "**/*.MD"],
            allow_empty = True,
        ),
        visibility = [
            "//" + native.package_name() + ":__subpackages__",
            "//plugins:__subpackages__",
        ],
    )
    native.label_flag(
        name = _FLAG,
        build_setting_default = ":" + _NONE,
        visibility = ["//visibility:public"],
    )
    _overlay_check_finalizer(
        name = _CHECK,
        overlay = ":" + _FLAG,
    )

def _is_path(src):
    return type(src) == "string" and not src.startswith((":", "//", "@"))

def _overlay_target(path):
    name = path + ".overlay"
    if not native.existing_rule(name):
        _overlay_file(
            name = name,
            src = path,
            path = path,
            overlay = ":" + _FLAG,
            check = ":" + _CHECK,
        )
    return ":" + name

def _expand_filegroup(src):
    """Returns the files of a same-package `filegroup` label, or None."""
    if type(src) != "string" or not src.startswith(":"):
        return None
    rule = native.existing_rule(src[1:])
    if not rule or rule["kind"] != "filegroup" or rule.get("output_group"):
        return None
    package_prefix = "//" + native.package_name() + ":"
    files = []
    for label in rule["srcs"]:
        label = str(label).removeprefix("@@").removeprefix(package_prefix)
        if not label.startswith(":"):
            return None
        rel = label[1:]
        files.append(label if native.existing_rule(rel) else rel)
    return files

# buildifier: disable=unnamed-macro
def overlay_srcs(srcs):
    """Routes each package-relative path in `srcs` through the package overlay.

    Args:
        srcs: A list of labels or paths, or a `select`, which is left as is.

    Returns:
        `srcs` unchanged if the package is not opted in, otherwise with each
        plain path replaced by its `<path>.overlay` target.
    """
    if not _enabled() or type(srcs) != "list":
        return srcs
    _ensure_package_targets()
    result = []
    for src in srcs:
        expanded = _expand_filegroup(src)
        if expanded == None:
            result.append(_overlay_target(src) if _is_path(src) else src)
            continue
        name = src[1:] + ".overlay"
        if not native.existing_rule(name):
            _overlay_filegroup(
                name = name,
                srcs = [
                    _overlay_target(item) if _is_path(item) else item
                    for item in expanded
                ],
                base = src,
            )
        result.append(":" + name)
    return result

# buildifier: disable=unnamed-macro
def overlay_src(src):
    """Like `overlay_srcs`, for a single label or path.

    Args:
        src: A label or path, or None.

    Returns:
        `src`, or its `<path>.overlay` target if the package is opted in.
    """
    if not _enabled() or not _is_path(src):
        return src
    _ensure_package_targets()
    return _overlay_target(src)

def overlay_includes(includes):
    """Places `overlay_tree/<dir>` ahead of each relative include directory.

    Args:
        includes: A list of include directories, or a `select`, which is left
            as is.

    Returns:
        `includes`, with the staged include directories added if the package is
        opted in.
    """
    if not _enabled() or type(includes) != "list":
        return includes
    result = []
    for include in includes:
        if not include.startswith("/"):
            result.append(_TREE + "/" + include)
        result.append(include)
    return result
