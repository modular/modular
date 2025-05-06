# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo doc %s | FileCheck %s


trait TraitWithAlias:
    # CHECK-DAG: "summary": "It's a trait, with an alias."
    """It's a trait, with an alias."""

    # CHECK-DAG: "traits"
    # CHECK-DAG: "aliases"
    # CHECK-DAG: "kind": "alias",
    # CHECK-DAG: "name": "N",
    # CHECK-DAG: "summary": "This is the alias."
    alias N: Int
    """This is the alias."""
