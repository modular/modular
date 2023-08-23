# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics -split-input-file %s

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

# // -----

##===----------------------------------------------------------------------===##
# Closure Captures
##===----------------------------------------------------------------------===##

fn captures_closure(x:Int):
   fn closure1(y:Int) escaping -> Int:
      return x + y
   fn closure2(y: Int) -> Int:
      return x * y
   # expected-error @below {{TODO: Cannot capture a signature type that escapes until new closures are turned on.}}
   # expected-error @below {{TODO: Cannot capture a signature type that captures until new closures are turned on.}}
   fn closure3(y:Int) escaping -> Int:
      let z = closure1(x)
      let w = closure2(z)
      return w
