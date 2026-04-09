# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: kgen-doc --diagnostic-format json %s 2>&1 | FileCheck %s
# CHECK: "line":[[@LINE+2]]{{.*}}"message":"doc string summary{{.*}}"
def x():
    """foo."""
    pass
