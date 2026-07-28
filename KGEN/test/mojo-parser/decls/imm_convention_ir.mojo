# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# 'imm' and 'read' are the same convention: both lower to an imm-origin
# read_mem reference. (The 'read' spelling additionally warns; warnings go to
# stderr and are not FileChecked here.)

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK: lit.fn @"imm_spelling(::String)"[imm *"x`"](%x: !lit.ref<!String, imm *"x`"> read_mem)
def imm_spelling(imm x: String) -> Int:
    return 1


# CHECK: lit.fn @"read_spelling(::String)"[imm *"x`"](%x: !lit.ref<!String, imm *"x`"> read_mem)
def read_spelling(read x: String) -> Int:
    return 1
