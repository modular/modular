//===- KGEN/KGENPasses.h --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENPASSES_H
#define KGEN_KGENPASSES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
class ModuleOp;
namespace func {
class FuncDialect;
} // namespace func
namespace LLVM {
class LLVMDialect;
} // namespace LLVM
} // namespace mlir

namespace M::KGEN {
class KGENDialect;
class KernelOp;

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createLowerHLKGENPass();
std::unique_ptr<mlir::Pass> createConvertKGENToLLVMPass();
std::unique_ptr<mlir::Pass> createConvertPOPToLLVMPass();

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "KGEN/KGENPasses.h.inc"

} // namespace M::KGEN

#endif // KGEN_KGENPASSES_H
