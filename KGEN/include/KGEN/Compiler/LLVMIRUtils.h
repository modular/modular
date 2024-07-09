//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_OBJECTCOMPILER_LLVMIRUTILS_H
#define KGEN_OBJECTCOMPILER_LLVMIRUTILS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/IR/Module.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// LLVMModuleAndContext
//===----------------------------------------------------------------------===//

/// A pair of an LLVM module and the LLVM context that holds ownership of the
/// objects. This is a useful class for parallelizing LLVM and managing
/// ownership of LLVM instances.
class LLVMModuleAndContext {
public:
  /// Expose the underlying LLVM context to create the module. This is the only
  /// way to access the LLVM context to prevent accidental sharing.
  ErrorOrSuccess create(
      function_ref<ErrorOr<std::unique_ptr<llvm::Module>>(llvm::LLVMContext &)>
          createModule);

  llvm::Module &operator*() { return *module; }
  llvm::Module *operator->() { return module.get(); }

private:
  /// LLVM context stored in a unique pointer so that we can move this type.
  std::unique_ptr<llvm::LLVMContext> ctx =
      std::make_unique<llvm::LLVMContext>();
  /// The paired LLVM module.
  std::unique_ptr<llvm::Module> module;
};

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

using LLVMSplitProcessFn =
    function_ref<void(LLVMModuleAndContext, std::optional<int64_t>)>;

/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
void splitPerExported(LLVMModuleAndContext module,
                      LLVMSplitProcessFn processFn);

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void splitPerFunction(LLVMModuleAndContext, size_t parallelismLevel,
                      LLVMSplitProcessFn processFn,
                      function_ref<void()> batchFn);

} // namespace M::KGEN

#endif // KGEN_OBJECTCOMPILER_LLVMIRUTILS_H
