# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: not mojo doc -strip-file-prefix=%S %s 2>&1 | FileCheck %s
# CHECK: {{^}}mojo_doc_strip_file_prefix.mojo:11:5: error
fn main():
    4 = "hello"
