# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s

struct StringNoCopy:
   var size: __mlir_type.index
   fn __init__(inout self):
      pass

   fn __del__(owned self):
      pass

fn makes_escaping_closurenocopy(m: StringNoCopy):
   fn myclosure() escaping -> StringNoCopy:
      # expected-error @+1 {{value of type 'StringNoCopy' cannot be copied into its destination}}
      return m
