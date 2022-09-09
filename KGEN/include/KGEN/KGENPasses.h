//===- KGEN/KGENPasses.h --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPASSES_H
#define KGEN_KGENPASSES_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
class ModuleOp;
namespace LLVM {
class LLVMDialect;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace M::KGEN {
class KGENDialect;
class MetaDialect;
class POPDialect;
class FuncOp;

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createLowerHLKGENPass();
std::unique_ptr<mlir::Pass>
createConvertKGENToLLVMPass(ArrayRef<StringRef> breakUpStructs = {},
                            ArrayRef<StringRef> emitCWrappers = {},
                            bool emitOpaqueWrappers = false);
std::unique_ptr<mlir::Pass> createConvertPOPToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertSCFToLLVMPass();
std::unique_ptr<mlir::Pass> createElaborateGeneratorsPass();
std::unique_ptr<mlir::Pass> createLowerZAPToPOPPass();

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "KGEN/KGENPasses.h.inc"

} // namespace M::KGEN

#endif // KGEN_KGENPASSES_H
