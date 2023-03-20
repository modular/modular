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
} // namespace M::LLCL

namespace M::KGEN {
/// This populates the pre-elaboration phase passes of the KGEN compiler. The
/// distribution format of a KGEN library is essentially what comes just before
/// elaboration because the parameter system allows significant extension.
void populateGenerateLibraryFilePasses(mlir::PassManager &pm);

/// This populates the passes to produce a fully concrete KGEN module. That
/// means it runs pre-elaboration, elaboration, and then the post-elaboration
/// cleanup passes. Its purpose is to populate the passes used to produce the
/// format that we will end up using to produce an object file.
void populateElaborateModulePasses(
    mlir::PassManager &pm, LLCL::Runtime &runtime, TargetInfoAttr target,
    const ElaborateGeneratorsOptions &elaborateOptions);
} // namespace M::KGEN

#endif // KGEN_COMPILER_H
