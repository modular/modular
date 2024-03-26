//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_CACHEPASSES_H
#define CACHE_CACHEPASSES_H

#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace mlir {
class ModuleOp;
class OpPassManager;
namespace LLVM {
class LLVMDialect;
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace M::LLCL {
class Runtime;
}

namespace M::Cache {
class CacheDialect;

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Cache/CachePasses/CachePasses.h.inc"

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Create an instance of the pass with the given LLCL::Runtime.
std::unique_ptr<mlir::Pass> createInflateSymbolsPass(LLCL::Runtime &rt);
std::unique_ptr<mlir::Pass> createInflateConstantsPass(LLCL::Runtime &rt);

/// Register the cache passes - their constructors require the LLCL::Runtime
/// provided.
void registerCachePasses(LLCL::Runtime &rt);

} // namespace M::Cache

#endif // CACHE_CACHEPASSES_H
