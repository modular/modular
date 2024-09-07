//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef KGEN_COMPILER_PIPELINE_PIPELINE_H
#define KGEN_COMPILER_PIPELINE_PIPELINE_H

#include "KGEN/Compiler/KGENCompiler.h"
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
                                  const CompilationOptions &options,
                                  KGENCompiler::StartPipelineAt startAt =
                                      KGENCompiler::StartPipelineAt::Beginning);

//===----------------------------------------------------------------------===//
// ElaborateModulePipeline
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. That
/// means it runs the elaborator and any dependent passes.
void buildElaborateModulePipeline(mlir::PassManager &pm, TargetInfoAttr target,
                                  const CompilationOptions &options,
                                  ElaboratorCompileAsmFn compileAsmFn);

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

/// This populates the passes to produce a fully concrete KGEN module. It's the
/// equivalent of the `buildElaborateModulePipeline` function, but with common
/// defaults for elaboration handlers.
void populateElaborateModulePasses(mlir::PassManager &pm, TargetInfoAttr target,
                                   const CompilationOptions &options);

//===----------------------------------------------------------------------===//
// PostElaborationPipeline
//===----------------------------------------------------------------------===//

/// This populates the post-elaboration optimization and simplification passes.
/// These passes are intended to run immediately after the elaborator.
void buildPostElaborationPipeline(mlir::PassManager &pm,
                                  const CompilationOptions &options);

} // namespace KGEN
} // namespace M

#endif // KGEN_COMPILER_PIPELINE_PIPELINE_H
