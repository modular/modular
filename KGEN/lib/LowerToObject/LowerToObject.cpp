//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "KGEN/CompilationOptions.h"
#include "LowerToObjectImpl.h"
#include "Support/ErrorOr.h"
#include "Support/TempFile.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Vectorize.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "lower-to-object"

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

/// Given a module operation, return its exported symbols.
static DenseSet<StringAttr> getExportedSymbols(ModuleOp module) {
  DenseSet<StringAttr> exportedSymbols;
  for (auto e : module.getOps<ExportOp>())
    for (auto sym : e.getExports().getAsRange<FlatSymbolRefAttr>())
      exportedSymbols.insert(sym.getAttr());
  return exportedSymbols;
}

ErrorOr<ObjectCompiler>
ObjectCompiler::create(LLCL::Runtime &runtime, StringRef basePath,
                       SymbolTable &symtab, const CompilationOptions &options) {
  return create(runtime, basePath, symtab,
                getExportedSymbols(cast<ModuleOp>(symtab.getOp())), options);
}

ErrorOr<ObjectCompiler>
ObjectCompiler::create(LLCL::Runtime &runtime, StringRef basePath,
                       SymbolTable &symtab, DenseSet<StringAttr> exports,
                       const CompilationOptions &options) {
  auto transformCache = Cache::getDefaultBackendChain(
      runtime, (std::filesystem::path(basePath.str()) / "transform").string());
  if (failed(transformCache))
    return transformCache.takeError();
  return ObjectCompiler(runtime, symtab, std::move(exports),
                        std::move(*transformCache), options);
}

ObjectCompiler::ObjectCompiler(
    LLCL::Runtime &runtime, SymbolTable &symtab, DenseSet<StringAttr> exports,
    LLCL::RCRef<Cache::BlobCacheBackend> transformCache,
    const CompilationOptions &options)
    : transformCache(
          decltype(this->transformCache)::create(std::move(transformCache))),
      runtime(runtime), module(cast<ModuleOp>(symtab.getOp())), symtab(symtab),
      exportedSymbols(std::move(exports)), options(options) {
  // Register types used during async compilation.
  LLCL::AsyncValue::registerTypes<Cache::BufferRef>();
}

//===----------------------------------------------------------------------===//
// populateFunctionPassManager
//===----------------------------------------------------------------------===//

void populateFunctionPassManager(
    llvm::legacy::FunctionPassManager &functionPassManager,
    llvm::TargetMachine &targetMachine) {
  unsigned OptLevel = targetMachine.getOptLevel();

  if (OptLevel == 0)
    return;

  functionPassManager.add(llvm::createTypeBasedAAWrapperPass());
  functionPassManager.add(llvm::createScopedNoAliasAAWrapperPass());

  // Lower llvm.expect to metadata before attempting transforms.
  // Compare/branch metadata may alter the behavior of passes like SimplifyCFG.
  functionPassManager.add(llvm::createCFGSimplificationPass());
  functionPassManager.add(llvm::createSROAPass());
  functionPassManager.add(llvm::createEarlyCSEPass());
}

//===----------------------------------------------------------------------===//
// populateModulePassManager
//===----------------------------------------------------------------------===//

void populateModulePassManager(llvm::legacy::PassManager &modulePassManager,
                               llvm::TargetMachine &targetMachine) {
  modulePassManager.add(llvm::createAnnotation2MetadataLegacyPass());

  // Allow forcing function attributes as a debugging and tuning aid.
  modulePassManager.add(llvm::createForceFunctionAttrsLegacyPass());

  // If all optimizations are disabled, just run the always-inline pass and,
  // if enabled, the function merging pass.
  unsigned OptLevel = targetMachine.getOptLevel();
  if (OptLevel == 0) {
    modulePassManager.add(llvm::createFunctionInliningPass(
        targetMachine.getOptLevel(), 0, false));
    return;
  }

  modulePassManager.add(llvm::createTypeBasedAAWrapperPass());
  modulePassManager.add(llvm::createScopedNoAliasAAWrapperPass());

  // Infer attributes about declarations if possible.
  modulePassManager.add(llvm::createInferFunctionAttrsLegacyPass());

  if (OptLevel > 2)
    modulePassManager.add(llvm::createCallSiteSplittingPass());

  modulePassManager.add(llvm::createIPSCCPPass()); // IP SCCP
  modulePassManager.add(llvm::createCalledValuePropagationPass());

  modulePassManager.add(
      llvm::createGlobalOptimizerPass()); // Optimize out global vars
  // Promote any localized global vars.
  modulePassManager.add(llvm::createPromoteMemoryToRegisterPass());

  modulePassManager.add(
      llvm::createDeadArgEliminationPass()); // Dead argument elimination

  modulePassManager.add(
      llvm::createInstructionCombiningPass()); // Clean up after IPCP & DAE
  modulePassManager.add(createCFGSimplificationPass(
      llvm::SimplifyCFGOptions().convertSwitchRangeToICmp(
          true))); // Clean up after IPCP & DAE

  // We add a module alias analysis pass here. In part due to bugs in the
  // analysis infrastructure this "works" in that the analysis stays alive
  // for the entire SCC pass run below.
  modulePassManager.add(llvm::createGlobalsAAWrapperPass());

  // Start of CallGraph SCC passes.
  modulePassManager.add(
      llvm::createFunctionInliningPass(OptLevel, /*SizeOptLevel=*/0,
                                       /*DisableInlineHotCallSite=*/false));

  modulePassManager.add(llvm::createPostOrderFunctionAttrsLegacyPass());

  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  modulePassManager.add(llvm::createSROAPass());
  modulePassManager.add(llvm::createEarlyCSEPass(
      true /* Enable mem-ssa. */)); // Catch trivial redundancies

  if (OptLevel > 1) {
    // Speculative execution if the target has divergent branches; otherwise
    // nop.
    modulePassManager.add(
        llvm::createSpeculativeExecutionIfHasBranchDivergencePass());

    modulePassManager.add(llvm::createJumpThreadingPass()); // Thread jumps.
    modulePassManager.add(
        llvm::createCorrelatedValuePropagationPass()); // Propagate conditionals
  }
  modulePassManager.add(createCFGSimplificationPass(
      llvm::SimplifyCFGOptions().convertSwitchRangeToICmp(
          true))); // Merge & remove BBs
  // Combine silly seq's
  modulePassManager.add(llvm::createInstructionCombiningPass());
  modulePassManager.add(llvm::createLibCallsShrinkWrapPass());

  if (OptLevel > 1)
    modulePassManager.add(
        llvm::createTailCallEliminationPass()); // Eliminate tail calls
  modulePassManager.add(createCFGSimplificationPass(
      llvm::SimplifyCFGOptions().convertSwitchRangeToICmp(
          true))); // Merge & remove BBs
  modulePassManager.add(
      llvm::createReassociatePass()); // Reassociate expressions

  // Begin the loop pass pipeline.

  // The simple loop unswitch pass relies on separate cleanup passes. Schedule
  // them first so when we re-process a loop they run before other loop
  // passes.
  modulePassManager.add(llvm::createLoopInstSimplifyPass());
  modulePassManager.add(llvm::createLoopSimplifyCFGPass());

  // Try to remove as much code from the loop header as possible,
  // to reduce amount of IR that will have to be duplicated. However,
  // do not perform speculative hoisting the first time as LICM
  // will destroy metadata that may not need to be destroyed if run
  // after loop rotation.
  // TODO: Investigate promotion cap for O1.
  modulePassManager.add(
      llvm::createLICMPass(/*LicmMssaOptCap=*/100,
                           /*LicmMssaNoAccForPromotionCap=*/250,
                           /*AllowSpeculation=*/false));
  // Rotate Loop - disable header duplication at -Oz
  modulePassManager.add(llvm::createLoopRotatePass(/*MaxHeaderSize=*/-1,
                                                   /*PrepareForLTO=*/false));
  modulePassManager.add(
      llvm::createLICMPass(/*LicmMssaOptCap=*/100,
                           /*LicmMssaNoAccForPromotionCap=*/250,
                           /*AllowSpeculation=*/true));
  modulePassManager.add(
      llvm::createSimpleLoopUnswitchLegacyPass(OptLevel == 3));
  // FIXME: We break the loop pass pipeline here in order to do full
  // simplifycfg. Eventually loop-simplifycfg should be enhanced to replace the
  // need for this.
  modulePassManager.add(createCFGSimplificationPass(
      llvm::SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
  modulePassManager.add(llvm::createInstructionCombiningPass());
  // We resume loop passes creating a second loop pipeline here.
  modulePassManager.add(
      llvm::createLoopIdiomPass()); // Recognize idioms like memset.
  modulePassManager.add(
      llvm::createIndVarSimplifyPass());                 // Canonicalize indvars
  modulePassManager.add(llvm::createLoopDeletionPass()); // Delete dead loops

  // Unroll small loops and perform peeling.
  modulePassManager.add(
      llvm::createSimpleLoopUnrollPass(OptLevel,
                                       /*OnlyWhenForced=*/false,
                                       /*ForgetAllSCEV=*/false));
  // This ends the loop pass pipelines.

  // Break up allocas that may now be splittable after loop unrolling.
  modulePassManager.add(llvm::createSROAPass());

  if (OptLevel > 1) {
    modulePassManager.add(
        llvm::createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds
    modulePassManager.add(llvm::createGVNPass(
        /*NoMemDepAnalysis=*/false)); // Remove redundancies
  }
  modulePassManager.add(llvm::createSCCPPass()); // Constant prop with SCCP

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  modulePassManager.add(
      llvm::createBitTrackingDCEPass()); // Delete dead bit computations

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  modulePassManager.add(llvm::createInstructionCombiningPass());
  if (OptLevel > 1) {
    modulePassManager.add(llvm::createJumpThreadingPass()); // Thread jumps
    modulePassManager.add(llvm::createCorrelatedValuePropagationPass());
  }
  modulePassManager.add(
      llvm::createAggressiveDCEPass()); // Delete dead instructions

  modulePassManager.add(
      llvm::createMemCpyOptPass()); // Remove memcpy / form memset
  if (OptLevel > 1) {
    modulePassManager.add(
        llvm::createDeadStoreEliminationPass()); // Delete dead stores
    modulePassManager.add(
        llvm::createLICMPass(/*LicmMssaOptCap=*/100,
                             /*LicmMssaNoAccForPromotionCap=*/250,
                             /*AllowSpeculation=*/true));
  }

  // Merge & remove BBs and sink & hoist common instructions.
  modulePassManager.add(createCFGSimplificationPass(
      llvm::SimplifyCFGOptions().hoistCommonInsts(true).sinkCommonInsts(true)));
  // Clean up after everything.
  modulePassManager.add(llvm::createInstructionCombiningPass());

  // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
  // pass manager that we are specifically trying to avoid. To prevent this
  // we must insert a no-op module pass to reset the pass manager.
  modulePassManager.add(llvm::createBarrierNoopPass());

  if (OptLevel > 1)
    // Remove avail extern fns and globals definitions if we aren't
    // compiling an object file for later LTO. For LTO we want to preserve
    // these so they are eligible for inlining at link-time. Note if they
    // are unreferenced they will be removed by GlobalDCE later, so
    // this only impacts referenced available externally globals.
    // Eventually they will be suppressed during codegen, but eliminating
    // here enables more opportunity for GlobalDCE as it may make
    // globals referenced by available external functions dead
    // and saves running remaining passes on the eliminated functions.
    modulePassManager.add(llvm::createEliminateAvailableExternallyPass());

  modulePassManager.add(llvm::createReversePostOrderFunctionAttrsPass());

  // The inliner performs some kind of dead code elimination as it goes,
  // but there are cases that are not really caught by it. We might
  // at some point consider teaching the inliner about them, but it
  // is OK for now to run GlobalOpt + GlobalDCE in tandem as their
  // benefits generally outweight the cost, making the whole pipeline
  // faster.
  modulePassManager.add(llvm::createGlobalOptimizerPass());
  modulePassManager.add(llvm::createGlobalDCEPass());

  // We add a fresh GlobalsModRef run at this point. This is particularly
  // useful as the above will have inlined, DCE'ed, and function-attr
  // propagated everything. We should at this point have a reasonably minimal
  // and richly annotated call graph. By computing aliasing and mod/ref
  // information for all local globals here, the late loop passes and notably
  // the vectorizer will be able to use them to help recognize vectorizable
  // memory operations.
  //
  // Note that this relies on a bug in the pass manager which preserves
  // a module analysis into a function pass pipeline (and throughout it) so
  // long as the first function pass doesn't invalidate the module analysis.
  // Thus both Float2Int and LoopRotate have to preserve AliasAnalysis for
  // this to work. Fortunately, it is trivial to preserve AliasAnalysis
  // (doing nothing preserves it as it is required to be conservatively
  // correct in the face of IR changes).
  modulePassManager.add(llvm::createGlobalsAAWrapperPass());

  modulePassManager.add(llvm::createFloat2IntPass());
  modulePassManager.add(llvm::createLowerConstantIntrinsicsPass());

  // Re-rotate loops in all our loop nests. These may have fallout out of
  // rotated form due to GVN or other transformations, and the vectorizer relies
  // on the rotated form. Disable header duplication at -Oz.
  modulePassManager.add(llvm::createLoopRotatePass(-1, false));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependences
  // into separate loop that would otherwise inhibit vectorization.  This is
  // currently only performed for loops marked with the metadata
  // llvm.loop.distribute=true or when -enable-loop-distribute is specified.
  modulePassManager.add(llvm::createLoopDistributePass());

  modulePassManager.add(
      llvm::createLoopVectorizePass(/*InterleaveOnlyWhenForced=*/false,
                                    /*VectorizeOnlyWhenForced=*/false));

  // Eliminate loads by forwarding stores from the previous iteration to loads
  // of the current iteration.
  modulePassManager.add(llvm::createLoopLoadEliminationPass());

  // Cleanup after the loop optimization passes.
  modulePassManager.add(llvm::createInstructionCombiningPass());

  // Now that we've formed fast to execute loop structures, we do further
  // optimizations. These are run afterward as they might block doing complex
  // analyses and transforms such as what are needed for loop vectorization.

  // Cleanup after loop vectorization, etc. Simplification passes like CVP and
  // GVN, loop transforms, and others have already run, so it's now better to
  // convert to more optimized IR using more aggressive simplify CFG options.
  // The extra sinking transform can create larger basic blocks, so do this
  // before SLP vectorization.
  modulePassManager.add(
      createCFGSimplificationPass(llvm::SimplifyCFGOptions()
                                      .forwardSwitchCondToPhi(true)
                                      .convertSwitchRangeToICmp(true)
                                      .convertSwitchToLookupTable(true)
                                      .needCanonicalLoops(false)
                                      .hoistCommonInsts(true)
                                      .sinkCommonInsts(true)));

  // Enhance/cleanup vector code.
  modulePassManager.add(llvm::createVectorCombinePass());

  modulePassManager.add(llvm::createInstructionCombiningPass());

  // Unroll small loops
  modulePassManager.add(llvm::createLoopUnrollPass(OptLevel,
                                                   /*OnlyWhenForced=*/false,
                                                   /*ForgetAllSCEV=*/false));

  // LoopUnroll may generate some redundency to cleanup.
  modulePassManager.add(llvm::createInstructionCombiningPass());

  // Runtime unrolling will introduce runtime check in loop prologue. If the
  // unrolled loop is a inner loop, then the prologue will be inside the
  // outer loop. LICM pass can help to promote the runtime check out if the
  // checked value is loop invariant.
  modulePassManager.add(
      llvm::createLICMPass(/*LicmMssaOptCap=*/100,
                           /*LicmMssaNoAccForPromotionCap=*/250,
                           /*AllowSpeculation=*/true));

  modulePassManager.add(llvm::createWarnMissedTransformationsPass());

  // After vectorization and unrolling, assume intrinsics may tell us more
  // about pointer alignments.
  modulePassManager.add(llvm::createAlignmentFromAssumptionsPass());

  // FIXME: We shouldn't bother with this anymore.
  modulePassManager.add(
      llvm::createStripDeadPrototypesPass()); // Get rid of dead prototypes

  // GlobalOpt already deletes dead functions and globals, at -O2 try a
  // late pass of GlobalDCE.  It is capable of deleting dead cycles.
  if (OptLevel > 1) {
    modulePassManager.add(
        llvm::createGlobalDCEPass()); // Remove dead fns and globals.
    modulePassManager.add(
        llvm::createConstantMergePass()); // Merge dup global constants
  }

  // LoopSink pass sinks instructions hoisted by LICM, which serves as a
  // canonicalization pass that enables other optimizations. As a result,
  // LoopSink pass needs to be a very late IR pass to avoid undoing LICM
  // result too early.
  modulePassManager.add(llvm::createLoopSinkPass());
  // Get rid of LCSSA nodes.
  modulePassManager.add(llvm::createInstSimplifyLegacyPass());

  // This hoists/decomposes div/rem ops. It should run after other sink/hoist
  // passes to avoid re-sinking, but before SimplifyCFG because it can allow
  // flattening of blocks.
  modulePassManager.add(llvm::createDivRemPairsPass());

  // LoopSink (and other loop passes since the last simplifyCFG) might have
  // resulted in single-entry-single-exit or empty blocks. Clean up the CFG.
  modulePassManager.add(createCFGSimplificationPass(
      llvm::SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
}

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//

LogicalResult KGEN::compileLLVMToObject(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        llvm::raw_pwrite_stream &objStream,
                                        bool emitAssembly) {
  TimeTraceScope<> traceScope("compile-llvm-to-object", module.getName());
  module.setDataLayout(targetMachine.createDataLayout());

  llvm::legacy::PassManager passManager;
  llvm::legacy::FunctionPassManager functionPassManager(&module);

  // Set up the pass manager and populate it.
  populateFunctionPassManager(functionPassManager, targetMachine);
  populateModulePassManager(passManager, targetMachine);

  functionPassManager.doInitialization();
  functionPassManager.doFinalization();

  // Add passes to emit an object file.
  targetMachine.addPassesToEmitFile(passManager, objStream, nullptr,
                                    emitAssembly ? llvm::CGFT_AssemblyFile
                                                 : llvm::CGFT_ObjectFile);

  // Run the pass manager to compile the module.
  for (auto &fun : module)
    functionPassManager.run(fun);

  passManager.run(module);

  return success();
}

//===----------------------------------------------------------------------===//
// createTargetMachine
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
KGEN::createTargetMachine(TargetInfoAttr targetInfo,
                          const CompilationOptions &options, bool isJIT) {
  { // TODO: remove this once we have more cross-compilation capability.
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    assert(targetInfo.getTripleStr() == targetTriple &&
           "TODO: target info must match host for now");
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm

  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      targetInfo.getTripleStr(), errorMessage);
  if (!target)
    return Error("no target exists for '" + targetInfo.getTripleStr() +
                 "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetInfo.getTripleStr(), targetInfo.getCpu(), targetInfo.getFeatures(),
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/std::nullopt, /*OL=*/options.getCodeGenOptLevel(), /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}
