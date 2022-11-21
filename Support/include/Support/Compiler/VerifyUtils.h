//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains helpers to write MLIR op verifiers.
//
//===----------------------------------------------------------------------===//

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {

/// Check that the op has the expected result types.
LogicalResult checkResultTypes(Operation *op, TypeRange expectedTypes);

} // namespace M
