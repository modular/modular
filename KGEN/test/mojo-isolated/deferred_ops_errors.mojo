# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-error @below {{invalid MLIR attribute: `#kgen.deferred` can only be used for non-typed attributes}}
# expected-note @below {{attempting to parse: '#kgen.deferred 0 : index'}}
_ = __mlir_attr.`#kgen.deferred 0 : index`
