# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: mojo doc --diagnostic-format json %s 2>&1 | FileCheck %s
# CHECK: "line":[[@LINE+2]]{{.*}}"message":"doc string summary{{.*}}"
fn x():
    """foo."""
    pass
