//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// LLVM IR processing path for kgen-opt.
//
//===----------------------------------------------------------------------===//

#include "KGEN/tools/kgen-opt/LLVMDriver.h"

#include "KGEN/Compiler/LLVMOptimizationPipeline.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/CommonCLOptions.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllIR.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

enum class BCVersionNo : uint16_t {
  DEFAULT = 0,
  LLVM17 = 17,
  LLVM19 = 19,
  LLVM21 = 21,
};

enum OutputKind {
  OK_NoOutput,
  OK_OutputAssembly,
  OK_OutputBitcode,
};

//===----------------------------------------------------------------------===//
// CL options (owned by this translation unit)
//===----------------------------------------------------------------------===//

struct LLVMCLOptions {
  cl::OptionCategory cat{"LLVM IR optimizer options"};

  cl::opt<std::string> passPipeline{
      "passes",
      cl::desc(
          "A textual LLVM pass pipeline (same syntax as opt -passes=...). "
          "Mutually exclusive with -O0/-O1/-O2/-O3.\n"
          "Available KGEN module passes:\n"
          "  kgen-metal-air           Transform IR to Apple AIR for Metal GPU\n"
          "  kgen-pointer-rewriter    Rewrite opaque pointers to typed "
          "pointers\n"
          "  kgen-metal-verifier      Reject IR with float types wider than "
          "f32\n"
          "  kgen-metal-rewrite-di    Rewrite DebugInfo for Metal/Instruments\n"
          "  kgen-llvmir-downgrade    Downgrade IR for older LLVM backends\n"
          "  kgen-set-function-attrs  Set function attributes for compilation\n"
          "Available KGEN function passes:\n"
          "  kgen-instruction-rewrite Rewrite unsupported "
          "intrinsics/instructions"),
      cl::value_desc("pipeline"), cl::cat(cat)};

  cl::opt<bool> optLevelO0{
      "O0", cl::desc("Optimization level 0. Similar to mojo -O0."),
      cl::cat(cat)};
  cl::opt<bool> optLevelO1{
      "O1", cl::desc("Optimization level 1. Similar to mojo -O1."),
      cl::cat(cat)};
  cl::opt<bool> optLevelO2{
      "O2", cl::desc("Optimization level 2. Similar to mojo -O2."),
      cl::cat(cat)};
  cl::opt<bool> optLevelO3{
      "O3", cl::desc("Optimization level 3. Similar to mojo -O3."),
      cl::cat(cat)};

  cl::opt<std::string> targetTriple{
      "mtriple", cl::desc("Override target triple for module"), cl::cat(cat)};

  cl::opt<std::string> dataLayout{
      "data-layout", cl::desc("Data layout string to use"),
      cl::value_desc("layout-string"), cl::cat(cat)};

  cl::opt<bool> noOutput{
      "disable-output",
      cl::desc("Do not write result bitcode file (LLVM IR path)."), cl::Hidden,
      cl::cat(cat)};

  cl::opt<bool> outputAssembly{
      "S", cl::desc("Write output as LLVM assembly (LLVM IR path)."),
      cl::cat(cat)};

  cl::opt<unsigned> codeGenOptLevel{
      "codegen-opt-level",
      cl::desc("Override optimization level for codegen hooks "
               "(legacy PM only, LLVM IR path)."),
      cl::cat(cat)};

  cl::opt<bool> disableOptimizationPasses{
      "disable-optimization-passes",
      cl::desc("Disable optimization passes and print input module "
               "(LLVM IR path). Useful to test Bitcode Writer."),
      cl::cat(cat)};

  cl::opt<BCVersionNo> outputBCVersion{
      "output-bc-version",
      cl::desc("Output bitcode LLVM version (LLVM IR path)."),
      cl::Hidden,
      cl::values(
          clEnumValN(BCVersionNo::DEFAULT, "default",
                     "Default bitcode version, no downgrading."),
          clEnumValN(BCVersionNo::LLVM17, "llvm17", "Bitcode version 17."),
          clEnumValN(BCVersionNo::LLVM19, "llvm19", "Bitcode version 19."),
          clEnumValN(BCVersionNo::LLVM21, "llvm21", "Bitcode version 21.")),
      cl::init(BCVersionNo::DEFAULT),
      cl::cat(cat)};

  cl::opt<bool> downgradeIR{
      "downgrade-llvm-ir",
      cl::desc("Run LLVMIRDowngrade pass for older LLVM backends "
               "(LLVM IR path)."),
      cl::cat(cat)};
};

/// Lazily constructed singleton that owns all LLVM-path CL options.
/// Construction registers the options with the global CL state.
LLVMCLOptions &getCLOptions() {
  static LLVMCLOptions opts;
  return opts;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Custom hack to handle Metal that is not supported in upstream:
// replace air64 with arm64.
std::string fixTargetTriple(std::string triple) {
  if (triple.find("air64") != std::string::npos)
    triple.replace(triple.find("air64"), 5, "arm64");
  return triple;
}

CodeGenOptLevel getCodeGenOptLevel(const LLVMCLOptions &opts) {
  return static_cast<CodeGenOptLevel>(unsigned(opts.codeGenOptLevel));
}

ModulePassManager buildLLVMPipeline(PassBuilder &pb, const LLVMCLOptions &opts,
                                    Triple triple) {
  ModulePassManager mpm;

  if (!opts.passPipeline.empty()) {
    if (llvm::Error err = pb.parsePassPipeline(mpm, opts.passPipeline)) {
      errs() << "error: failed to parse pass pipeline '" << opts.passPipeline
             << "': " << toString(std::move(err)) << "\n";
      exit(1);
    }
    return mpm;
  }

  M::KGEN::CompilationOptions options(/*optimizationLevel=*/-1U);
  options.targetTriple = triple.str();
  if (opts.optLevelO0)
    options.optimizationLevel = 0;
  if (opts.optLevelO1)
    options.optimizationLevel = 1;
  if (opts.optLevelO2)
    options.optimizationLevel = 2;
  if (opts.optLevelO3)
    options.optimizationLevel = 3;

  if (options.optimizationLevel == -1U)
    llvm_unreachable(
        "Specify an optimization level (-O0/-O1/-O2/-O3) or a custom pipeline "
        "(-passes=...).");

  return M::KGEN::buildLLVMOptimizationPipeline(pb, options);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void M::KGEN::Tool::registerLLVMPathCLOptions() {
  // Trigger construction of all LLVM-path cl::opt objects.
  getCLOptions();
  // Register the target/CPU printer shown by --version.
  llvm::cl::AddExtraVersionPrinter(llvm::sys::printDefaultTargetAndDetectedCPU);
}

int M::KGEN::Tool::runLLVMPath(StringRef inputFile, StringRef outputFile) {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();
  InitializeNativeTarget();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetAsmPrinter();

  // Register legacy pass-manager passes used by codegen hooks and opt
  // pipelines.
  {
    PassRegistry &pr = *PassRegistry::getPassRegistry();
    initializeCore(pr);
    initializeScalarOpts(pr);
    initializeVectorization(pr);
    initializeIPO(pr);
    initializeAnalysis(pr);
    initializeTransformUtils(pr);
    initializeInstCombine(pr);
    initializeTarget(pr);
    initializeExpandIRInstsLegacyPassPass(pr);
    initializeExpandMemCmpLegacyPassPass(pr);
    initializeScalarizeMaskedMemIntrinLegacyPassPass(pr);
    initializeSelectOptimizePass(pr);
    initializeInlineAsmPreparePass(pr);
    initializeCodeGenPrepareLegacyPassPass(pr);
    initializeAtomicExpandLegacyPass(pr);
    initializeWinEHPreparePass(pr);
    initializeDwarfEHPrepareLegacyPassPass(pr);
    initializeSafeStackLegacyPassPass(pr);
    initializeSjLjEHPreparePass(pr);
    initializePreISelIntrinsicLoweringLegacyPassPass(pr);
    initializeGlobalMergePass(pr);
    initializeIndirectBrExpandLegacyPassPass(pr);
    initializeInterleavedLoadCombinePass(pr);
    initializeInterleavedAccessPass(pr);
    initializePostInlineEntryExitInstrumenterPass(pr);
    initializeUnreachableBlockElimLegacyPassPass(pr);
    initializeExpandReductionsPass(pr);
    initializeWasmEHPreparePass(pr);
    initializeWriteBitcodePassPass(pr);
    initializeReplaceWithVeclibLegacyPass(pr);
    initializeJMCInstrumenterPass(pr);
    initializeRuntimeLibraryInfoWrapperPass(pr);
    initializeLibcallLoweringInfoWrapperPass(pr);
  }

  LLVMCLOptions &clOpts = getCLOptions();

  LLVMContext context;
  SMDiagnostic err;

  auto setDataLayout = [&](StringRef irTriple,
                           StringRef irLayout) -> std::optional<std::string> {
    if (!clOpts.dataLayout.empty())
      return std::nullopt;
    if (!irLayout.empty())
      return std::nullopt;

    std::string tripleStr = clOpts.targetTriple.empty()
                                ? irTriple.str()
                                : Triple::normalize(clOpts.targetTriple);
    tripleStr = fixTargetTriple(tripleStr);
    if (tripleStr.empty())
      return std::nullopt;

    Expected<std::unique_ptr<TargetMachine>> expectedTM =
        codegen::createTargetMachineForTriple(tripleStr,
                                              getCodeGenOptLevel(clOpts));
    if (!expectedTM) {
      errs() << "kgen-opt: warning: failed to infer data layout: "
             << toString(expectedTM.takeError()) << "\n";
      return std::nullopt;
    }
    return (*expectedTM)->createDataLayout().getStringRepresentation();
  };

  std::unique_ptr<Module> module =
      parseIRFile(inputFile, err, context, ParserCallbacks(setDataLayout));
  if (!module) {
    err.print("kgen-opt", errs());
    return 1;
  }

  OutputKind outputKind = OK_NoOutput;
  if (!clOpts.noOutput)
    outputKind = clOpts.outputAssembly ? OK_OutputAssembly : OK_OutputBitcode;

  std::string outFile = outputFile.str();
  std::unique_ptr<ToolOutputFile> out;
  if (clOpts.noOutput) {
    if (!outFile.empty() && outFile != "-")
      errs() << "WARNING: -o is ignored when --disable-output is set.\n";
  } else {
    if (outFile.empty())
      outFile = "-";
    std::error_code ec;
    sys::fs::OpenFlags flags =
        clOpts.outputAssembly ? sys::fs::OF_TextWithCRLF : sys::fs::OF_None;
    out.reset(new ToolOutputFile(outFile, ec, flags));
    if (ec) {
      errs() << ec.message() << '\n';
      return 1;
    }
  }

  if (!clOpts.targetTriple.empty())
    module->setTargetTriple(Triple(Triple::normalize(clOpts.targetTriple)));

  const bool isMetalTriple = M::KGEN::isMetalTriple(module->getTargetTriple());
  Triple moduleTriple(fixTargetTriple(module->getTargetTriple().str()));
  TargetLibraryInfoImpl tlii(moduleTriple);
  std::string cpuStr, featuresStr;
  std::unique_ptr<TargetMachine> targetMachine;

  if (isMetalTriple || moduleTriple.getArch()) {
    cpuStr = codegen::getCPUStr();
    featuresStr = codegen::getFeaturesStr();
    Expected<std::unique_ptr<TargetMachine>> expectedTM =
        codegen::createTargetMachineForTriple(moduleTriple.str(),
                                              getCodeGenOptLevel(clOpts));
    if (llvm::Error e = expectedTM.takeError()) {
      errs() << "kgen-opt: WARNING: failed to create target machine for '"
             << moduleTriple.str() << "': " << toString(std::move(e)) << "\n";
    } else {
      targetMachine = std::move(*expectedTM);
    }
  } else if (moduleTriple.getArchName() != "unknown" &&
             moduleTriple.getArchName() != "") {
    errs() << "kgen-opt: unrecognized architecture '"
           << moduleTriple.getArchName() << "' provided.\n";
    return 1;
  }

  codegen::setFunctionAttributes(*module, cpuStr, featuresStr);

  PassInstrumentationCallbacks pic;
  PassBuilder pb(targetMachine.get(), PipelineTuningOptions(),
                 /*PGOOpt=*/std::nullopt, &pic);
  M::KGEN::registerKGENLLVMPasses(pb);

  ModulePassManager mpm;
  if (!clOpts.disableOptimizationPasses)
    mpm = buildLLVMPipeline(pb, clOpts, module->getTargetTriple());
  if (clOpts.downgradeIR)
    M::KGEN::addLLVMIRDowngradePass(mpm);

  switch (outputKind) {
  case OK_NoOutput:
    break;
  case OK_OutputAssembly:
    mpm.addPass(PrintModulePass(out->os(), "",
                                /*ShouldPreserveAssemblyUseListOrder=*/false,
                                /*EmitSummaryIndex=*/false));
    break;
  case OK_OutputBitcode:
    if (!isMetalTriple && clOpts.outputBCVersion == BCVersionNo::DEFAULT) {
      mpm.addPass(BitcodeWriterPass(out->os(),
                                    /*ShouldPreserveBitcodeUseListOrder=*/false,
                                    /*EmitSummaryIndex=*/false,
                                    /*EmitModuleHash=*/false));
    }
    break;
  }

  AAManager aa;
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;

  fam.registerPass([&] { return std::move(aa); });
  fam.registerPass([&] { return TargetLibraryAnalysis(tlii); });
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  mpm.run(*module, mam);

  if (!clOpts.downgradeIR && verifyModule(*module, &errs()))
    return 1;

  if (isMetalTriple && outputKind == OK_OutputBitcode) {
    M::KGEN::LLVM::WriteBitcode17ToFile(*module, out->os(), false, nullptr,
                                        false, nullptr);
  } else if (outputKind == OK_OutputBitcode) {
    switch (clOpts.outputBCVersion) {
    case BCVersionNo::LLVM17:
      M::KGEN::LLVM::WriteBitcode17ToFile(*module, out->os(), false, nullptr,
                                          false, nullptr);
      break;
    case BCVersionNo::LLVM19:
      M::KGEN::LLVM::WriteBitcode19ToFile(*module, out->os(), false, nullptr,
                                          false, nullptr);
      break;
    case BCVersionNo::LLVM21:
      M::KGEN::LLVM::WriteBitcode21ToFile(*module, out->os(), false, nullptr,
                                          false, nullptr);
      break;
    case BCVersionNo::DEFAULT:
      break;
    }
  }

  if (outputKind != OK_NoOutput)
    out->keep();
  return 0;
}
