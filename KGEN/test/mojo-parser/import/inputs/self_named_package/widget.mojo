# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# The file being compiled shares its name ('widget') with a package it imports.
# The import must resolve to the sibling 'widget' package, not to this file.
from widget import gadget

def main():
    gadget.run()
