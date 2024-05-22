//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_OBJECTCOMPILER_H
#define KGEN_OBJECTCOMPILER_H

#include "Cache/CachedTransform.h"
#include "KGEN/ToolCommon/CompilationOptions.h"

namespace llvm {
class Module;
class TargetMachine;
} // namespace llvm

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// optimizeLLVMModule
//===----------------------------------------------------------------------===//

/// Optimize the llvm module to prepare for codegen object file.
LogicalResult
optimizeLLVMModule(llvm::Module &module, llvm::TargetMachine &targetMachine,
                   CompilationOptions &options, LLCL::Runtime &runtime,
                   std::optional<size_t> moduleIdx = std::nullopt);

//===----------------------------------------------------------------------===//
// compileOptimizedLLVMToObjects
//===----------------------------------------------------------------------===//

/// Compile the given LLVM module to object files and return the async values
/// that contains the compiled object file.
/// isParLLC is true: split module into per function for parallel llc lowering
///                   and return multiple object files.
/// isParLLC is false: compile module without splitting into one object file.
SmallVector<LLCL::AnyAsyncValueRef> compileOptimizedLLVMToObjects(
    llvm::Module &module, mlir::Location loc, CompilationOptions &options,
    LLCL::Runtime &runtime, RCRef<Cache::TransformCache> transformCache,
    bool isParLLC, bool isJIT, bool emitAssembly = false,
    std::optional<size_t> moduleIdx = std::nullopt);

//===----------------------------------------------------------------------===//
// runLLVMOptPasses
//===----------------------------------------------------------------------===//

/// Run the llvm opt passes over `module` given `targetMachine`.
LogicalResult runLLVMOptPasses(llvm::Module &module,
                               llvm::TargetMachine &targetMachine,
                               const CompilationOptions &options,
                               LLCL::Runtime &runtime);

} // namespace M::KGEN

#endif // KGEN_OBJECTCOMPILER_H
