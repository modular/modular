//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_OBJECTCOMPILER_LLVMIRUTILS_H
#define KGEN_OBJECTCOMPILER_LLVMIRUTILS_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/IR/Module.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
SmallVector<std::unique_ptr<llvm::Module>>
splitPerExported(llvm::Module &module);

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void splitPerFunction(
    std::unique_ptr<llvm::Module> module, size_t parallelismLevel,
    function_ref<void(std::unique_ptr<llvm::Module>, int64_t idx, bool)>
        processFn);

} // namespace M::KGEN

#endif // KGEN_OBJECTCOMPILER_LLVMIRUTILS_H
