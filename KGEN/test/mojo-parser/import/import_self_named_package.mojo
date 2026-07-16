# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# The file being compiled ('widget.mojo') shares its name with a package
# ('widget/') sitting alongside it. `from widget import gadget` must resolve to
# that sibling package rather than recursing into the file being compiled
# (MOCO-1946). The two coexist: the compilation root keeps the symbol @widget,
# and the imported package is uniqued to @widget_0 (both report source name
# "widget").

# RUN: %parse-mojo-isolated %S/inputs/self_named_package/widget.mojo | FileCheck %s

# The compilation root keeps its own (un-uniqued) symbol name.
# CHECK: lit.file_module @widget attributes {sourceName = "widget"}

# The self-named import resolves into the sibling package - uniqued to
# @widget_0 - reaching its @gadget submodule, not back into @widget.
# CHECK: lit.call tail @widget_0::@gadget::@"run()"

# The imported package coexists with the root under a uniqued symbol name but
# the same source name.
# CHECK: lit.package @widget_0 attributes {sourceName = "widget"}
# CHECK: lit.file_module @gadget attributes {sourceName = "gadget"}
