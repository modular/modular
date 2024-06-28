//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Custom dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CUSTOM_TD
#define CUSTOM_TD

include "KGEN/CustomDialect/CustomDialect.td"

// This include is required as `mlir-tblgen --gen-op-defs` requires the `Op` class to exist.
// Since we do not define any operations in our dialect, we need to include it here.
include "mlir/IR/OpBase.td"

#endif // CUSTOM_TD
