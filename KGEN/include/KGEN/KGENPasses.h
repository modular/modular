//===- KGEN/KGENPasses.h --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPASSES_H
#define KGEN_KGENPASSES_H

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class RewritePatternSet;
class Pass;
} // namespace mlir

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createLowerHLKGENPass();
std::unique_ptr<mlir::Pass> createConvertKGENToLLVMPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering these passes.
#define GEN_PASS_REGISTRATION
#include "KGEN/KGENPasses.h.inc"

} // namespace M::KGEN

#endif // KGEN_KGENPASSES_H
