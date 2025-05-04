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
#include "Support/LogicalResult.h"

namespace M {

/// Check that the op has the expected operand types.
LogicalResult checkOperandTypes(Operation *op, TypeRange expectedTypes);

/// Check that the block with the given name has the expected argument types.
/// The given op is only used to emit any errors.
LogicalResult checkArgumentTypes(Operation *op, StringRef blockName,
                                 Block *block, TypeRange expectedTypes);

/// Check that the two type ranges with the given context label agree. The given
/// op is only used to emit any errors.
LogicalResult checkMatchingTypes(Operation *op, StringRef context,
                                 TypeRange actualTypes,
                                 TypeRange expectedTypes);

} // namespace M
