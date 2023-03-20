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

/// This elaborates all the generators in `theModule` and takes the module from
/// a just-parsed state to a state we can use to produce an object file. This
/// modifies the module in place. The granularity of this operation is tentative
/// and should be re-evaluated, we may end up in a place where we want to split
/// pre-elaboration, elaboration, and post-elaboration into explicit phases.
///
/// The purpose of this function is largely for cases where we don't want to add
/// additional options to the pass manager, such as when we're evaluating a
/// module in a JIT context.
LogicalResult
concretizeModule(mlir::PassManager &pm, ModuleOp theModule,
                 LLCL::Runtime &runtime, TargetInfoAttr target,
                 const ElaborateGeneratorsOptions &elaborateOptions);
} // namespace M::KGEN

#endif // KGEN_COMPILER_H
