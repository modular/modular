# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that the deprecated '__deinit__' destructor spelling is diagnosed with a
# warning and a fix-it to the canonical '__deinit__' spelling, while the
# canonical spelling itself is silent.
# RUN: %parse-mojo-isolated -verify-diagnostics %s

# Verify the fix-it text itself via the JSON diagnostic format.
# RUN: %parse-mojo-isolated --diagnostic-format json --use-mlir-diagnostics=false %s 2>&1 | FileCheck %s --check-prefix=FIXIT

# FIXIT: "diagnostic":{
# FIXIT-SAME: "fixIts":[{
# FIXIT-SAME: "text":"__deinit__"


struct UsesOldSpelling:
    var x: Int

    def __init__(out self):
        self.x = 0

    # expected-warning @+1 {{'__del__' is deprecated; use '__deinit__'}}
    def __del__(deinit self):
        pass


struct UsesNewSpelling:
    var x: Int

    def __init__(out self):
        self.x = 0

    def __deinit__(deinit self):
        pass
