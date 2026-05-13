# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mkdir -p %t.contains-dupe-dir
# RUN: mojo package %S/inputs/containsDupe -o %t.contains-dupe-dir/containsDupe.mojoc
# RUN: mojo -I %t.contains-dupe-dir %s 2>&1

# Ensure the identical wrapper defined in containsDupe
# is not pulled into this file module scope,
# which would cause name conflicts because of the duplicate
# closure defined here. TODO: dedupe struct wrappers like traits are today?


from containsDupe import *


def main() raises:
    def identical() {var} -> String:
        return "hello"

    # CHECK: hello
    consume(identical)
