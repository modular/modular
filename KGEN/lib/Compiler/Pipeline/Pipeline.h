//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef KGEN_COMPILER_PIPELINE_PIPELINE_H
#define KGEN_COMPILER_PIPELINE_PIPELINE_H

#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/LLVMForwardDecls.h"

namespace M {
namespace KGEN {

//===----------------------------------------------------------------------===//
// CHECKLITPipeline
//===----------------------------------------------------------------------===//

/// This populates the post-parser pipeline that checks and lowers source-level
/// LIT constructs.
void buildCheckLITPipeline(mlir::PassManager &pm,
                           const CompilationOptions &options);

//===----------------------------------------------------------------------===//
// GenerateLibraryPipeline
//===----------------------------------------------------------------------===//

/// This populates the pre-elaboration phase passes of the KGEN compiler. The
/// distribution format of a KGEN library is essentially what comes just before
/// elaboration because the parameter system allows significant extension.
void buildGenerateLibraryPipeline(mlir::PassManager &pm,
                                  const CompilationOptions &options);

//===----------------------------------------------------------------------===//
// ElaborateModulePipeline
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. That
/// means it runs the elaborator and any dependent passes.
void buildElaborateModulePipeline(mlir::PassManager &pm, TargetInfoAttr target,
                                  const CompilationOptions &options,
                                  ElaboratorCompileAsmFn compileAsmFn,
                                  PackageGenLibraryFn packageGenLibraryFn);

} // namespace KGEN
} // namespace M

#endif // KGEN_COMPILER_PIPELINE_PIPELINE_H
