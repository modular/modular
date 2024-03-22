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
void splitPerExported(
    llvm::Module &module,
    function_ref<void(llvm::Module &, int64_t idx)> processFn);

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void splitPerFunction(
    llvm::Module &module,
    function_ref<void(llvm::Module &, int64_t idx)> processFn);

} // namespace M::KGEN

#endif // KGEN_OBJECTCOMPILER_LLVMIRUTILS_H
