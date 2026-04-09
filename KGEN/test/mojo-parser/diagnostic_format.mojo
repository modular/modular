# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that --diagnostic-format json produces valid JSON with expected structure.
# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s --check-prefix=JSON

# Verify the JSON contains expected fields and structure.
# JSON: "diagnostic":{
# JSON-SAME: "file":
# JSON-SAME: "fixIts":
# JSON-SAME: "location":{
# JSON-SAME: "column":
# JSON-SAME: "line":
# JSON-SAME: "text":
# JSON-SAME: "kind":"error"
# JSON-SAME: "message":"use of unknown declaration 'unknown_identifier'"

# Test that default text format still works.
# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s --check-prefix=TEXT
# TEXT: error:

# Test that JSON format with MLIR diagnostics is an error.
# RUN: not %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=true %s 2>&1 | FileCheck %s --check-prefix=ERROR
# ERROR: error: --diagnostic-format=json is incompatible with --use-mlir-diagnostics=true


def main():
    _ = unknown_identifier
