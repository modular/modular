//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_H
#define KGEN_COMPILER_H

#include "KGEN/KGENPasses.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::LLCL {
class Runtime;
}

namespace M::KGEN {
/// This runs the pre-elaboration phase of the KGEN compiler. The distribution
/// format of a KGEN library is essentially what comes just before elaboration
/// because the parameter system allows significant extension. This function
/// modifies the module in-place.
LogicalResult generateLibraryFile(ModuleOp theModule);
/// This produces a fully concrete KGEN module. That means it runs
/// pre-elaboration, elaboration, and then the post-elaboration cleanup passes.
/// Its purpose is to produce the format that we will end up using to produce an
/// object file. This function modifies the module in-place.
LogicalResult
elaborateModule(ModuleOp theModule, LLCL::Runtime &runtime,
                const ElaborateGeneratorsOptions &elaborateOptions,
                SmallVectorImpl<std::string> &includedFiles);
} // namespace M::KGEN

#endif // KGEN_COMPILER_H
