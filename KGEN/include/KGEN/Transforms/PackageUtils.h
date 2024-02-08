//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMS_PACKAGEUTILS_H
#define KGEN_TRANSFORMS_PACKAGEUTILS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
/// Loads the serialized MLIR bytecode representing a pre-elaborated module in
/// `bytecodeAttr`, and prepare to link it into directly another module. Returns
/// the module if successful, or an error.
ErrorOr<OwningOpRef<ModuleOp>>
loadPreElaboratedModuleForLinking(DenseResourceElementsAttr bytecodeAttr);

/// Loads the serialized MLIR bytecode representing a pre-elaborated module in
/// `bytecodeAttr`, and prepare to link it into directly another module. Returns
/// the bytecode if successful, or an error.
ErrorOr<DenseResourceElementsAttr>
loadPreElaboratedBytecodeForLinking(DenseResourceElementsAttr bytecodeAttr);
} // namespace M::KGEN

#endif // KGEN_TRANSFORMS_PACKAGEUTILS_H
