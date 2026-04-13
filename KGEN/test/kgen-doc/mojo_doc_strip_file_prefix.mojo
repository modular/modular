# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: not kgen-doc -strip-file-prefix=%S %s 2>&1 | FileCheck %s
# Diagnostics from the parser go through SourceMgr, not the tool's error
# prefix, so the expected output starts directly with the filename (no
# "kgen-doc: error:" prefix).
# CHECK: {{^}}mojo_doc_strip_file_prefix.mojo:14:5: error
def main():
    4 = "hello"
