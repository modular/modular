//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_OBJECTCOMPILER_KGENTOLLVMPIPELINE_H
#define KGEN_COMPILER_OBJECTCOMPILER_KGENTOLLVMPIPELINE_H

#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/LLVMForwardDecls.h"

namespace M {
namespace KGEN {

//===----------------------------------------------------------------------===//
// buildLowerToLLVMPipeline
//===----------------------------------------------------------------------===//
/// Build the pass pipeline to convert post-elaboration KGEN IR to LLVM IR.
/// The pipeline runs the canonicalizer, the KGEN to LLVM conversion, a series
/// of LLVM lowerings, and the canonicalizer again.
void buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                              const LowerToLLVMOptions &options);

} // namespace KGEN
} // namespace M

#endif // KGEN_COMPILER_OBJECTCOMPILER_KGENTOLLVMPIPELINE_H
