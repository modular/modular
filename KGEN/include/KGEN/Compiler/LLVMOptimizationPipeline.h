//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMPASSESPIPELINE_H
#define KGEN_COMPILER_LLVMPASSESPIPELINE_H

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace M::KGEN {

class CompilationOptions;

/// Build a module pass pipeline for a given set of compilation options.
llvm::ModulePassManager
buildLLVMOptimizationPipeline(llvm::PassBuilder &passBuilder,
                              const CompilationOptions &options);

/// Add LLVMIRDowngradePass to the pass manager.
void addLLVMIRDowngradePass(llvm::ModulePassManager &mpm);

/// Register all custom LLVM passes with \p passBuilder so they are available by
/// name when using PassBuilder's pipeline parsing (e.g. the -passes option).
///
/// Registered module passes:
///   kgen-metal-air            - MetalAIRPass
///   kgen-pointer-rewriter     - PointerRewriter
///   kgen-metal-verifier       - MetalVerifierPass
///   kgen-metal-rewrite-di     - MetalRewriteDebugInfoPass
///   kgen-llvmir-downgrade     - LLVMIRDowngradePass
///   kgen-set-function-attrs   - SetFunctionAttributes
///
/// Registered function passes:
///   kgen-instruction-rewrite  - InstructionRewritePass
void registerKGENLLVMPasses(llvm::PassBuilder &passBuilder);

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMPASSESPIPELINE_H
