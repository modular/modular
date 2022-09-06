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
class KernelOp;

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createLowerHLKGENPass();
std::unique_ptr<mlir::Pass>
createConvertKGENToLLVMPass(ArrayRef<StringRef> breakUpStructs = {},
                            ArrayRef<StringRef> emitCWrappers = {});
std::unique_ptr<mlir::Pass> createConvertPOPToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertSCFToLLVMPass();
std::unique_ptr<mlir::Pass> createElaborateGeneratorsPass();

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "KGEN/KGENPasses.h.inc"

} // namespace M::KGEN

#endif // KGEN_KGENPASSES_H
