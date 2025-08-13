"""List of NIXL plugins."""

NIXL_PLUGINS = [
    "ucx",
    "ucx_verbs",
]

NIXL_PLUGINS_TARGETS = [
    "//AsyncRT:plugin_{}".format(plugin)
    for plugin in NIXL_PLUGINS
]

NIXL_PLUGINS_DATA = select({
    "//:linux_x86_64": NIXL_PLUGINS_TARGETS,
    "//conditions:default": [],
})
