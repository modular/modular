"""Traverse dependencies to collect Mojo information."""

load("@rules_mojo//mojo:providers.bzl", "MojoInfo")

def _collect_mojoinfo_aspect_impl(target, ctx):
    """Implementation of the aspect to collect MojoInfo."""

    # 1. Check if the current target itself provides MojoInfo
    if MojoInfo in target:
        return []

    transitive_import_paths = []
    transitive_mojopkgs = []

    # 2. Iterate over specified attributes (e.g., 'deps', 'data') to find dependencies
    #    The aspect definition below will specify which attributes to traverse.
    for attr_name in ["deps", "data"]:
        attr = getattr(ctx.rule.attr, attr_name, [])

        for dep in attr:
            if MojoInfo in dep:
                transitive_import_paths.append(dep[MojoInfo].import_paths)
                transitive_mojopkgs.append(dep[MojoInfo].mojopkgs)

    # Return a new MojoInfo provider with the aggregated mojopkgs.
    # This allows transitive collection.
    return [
        MojoInfo(
            import_paths = depset(transitive = transitive_import_paths),
            mojopkgs = depset(transitive = transitive_mojopkgs),
        ),
    ]

collect_mojoinfo_aspect = aspect(
    implementation = _collect_mojoinfo_aspect_impl,
    attr_aspects = [
        # Attributes of rules to traverse for dependencies.
        # This aspect will be applied to targets found in these attributes.
        "deps",
        "data",
    ],
)

def _collect_transitive_mojoinfo_impl(ctx):
    """
    Implementation for the rule that collects MojoInfo using an aspect
    """
    import_paths = []
    mojopkgs = []

    # The 'deps_to_scan' attribute has the 'collect_data_metadata_aspect' applied to it.
    # Each target listed in 'deps_to_scan' (and their relevant transitive dependencies
    # as defined by the aspect's attr_aspects) will have the aspect run on them.
    # The aspect propagates MojoInfo upwards.
    # So, we iterate through the direct targets provided in 'deps_to_scan'.
    # If they (or targets they depend on) provided MojoInfo via the aspect,
    # that provider will be available on them here.
    for dep in ctx.attr.deps_to_scan:
        if MojoInfo in dep:
            import_paths.append(dep[MojoInfo].import_paths)
            mojopkgs.append(dep[MojoInfo].mojopkgs)

    return [
        MojoInfo(
            import_paths = depset(transitive = import_paths),
            mojopkgs = depset(transitive = mojopkgs),
        ),
    ]

collect_transitive_mojoinfo = rule(
    implementation = _collect_transitive_mojoinfo_impl,
    attrs = {
        "deps_to_scan": attr.label_list(
            doc = "Dependencies to scan for MojoInfo",
            aspects = [collect_mojoinfo_aspect],
            mandatory = True,
        ),
    },
    doc = "Collect MojoInfo from dependencies",
)
