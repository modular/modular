//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// DEPRECATED: kgen-llvm-opt is deprecated.
// Use kgen-opt instead; it automatically selects the LLVM IR path for .ll and
// .bc files, and the MLIR path for .mlir files.
//
// The kgen-llvm-opt tool is similar to LLVM's opt tool. It supports two modes:
//
//  1. Full KGEN optimization pipeline: specify -O0/-O1/-O2/-O3 to run the
//     complete Mojo compilation pipeline for that optimization level.
//
//  2. Custom pass pipeline via -passes: specify an explicit pass pipeline using
//     LLVM's pass pipeline syntax (same as `opt -passes=...`). All custom KGEN
//     passes defined in KGEN/lib/Compiler/ObjectCompiler/LLVM/Transforms are
//     registered and available under the following names:
//       module passes:
//         kgen-metal-air            - MetalAIRPass
//         kgen-pointer-rewriter     - PointerRewriter
//         kgen-metal-verifier       - MetalVerifierPass
//         kgen-metal-rewrite-di     - MetalRewriteDebugInfoPass
//         kgen-llvmir-downgrade     - LLVMIRDowngradePass
//         kgen-set-function-attrs   - SetFunctionAttributes
//       function passes:
//         kgen-instruction-rewrite  - InstructionRewritePass
//
//     Example: kgen-llvm-opt -passes="kgen-metal-air,kgen-pointer-rewriter"
//     in.bc

#include "KGEN/Compiler/LLVMOptimizationPipeline.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/ToolCommon/CLOptions.h"
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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {
// Debug emission kind
enum class BCVersionNo : uint16_t {
  DEFAULT = 0,
  LLVM17 = 17,
  LLVM19 = 19,
  LLVM21 = 21,
};
struct CLOptions : public M::CLOptionsBase {
  CLOptions(int argc, char **argv, bool skipInitLLVM = true)
      : M::CLOptionsBase(argc, argv, options, skipInitLLVM) {}

  M::OptionsBase options;
  std::string inputFilename{"-"};
  std::string outputFilename{"-"};
  std::string passPipeline;
  bool optLevelO0 = false;
  bool optLevelO1 = false;
  bool optLevelO2 = false;
  bool optLevelO3 = false;
  std::string targetTriple;
  std::string targetAccelerator;
  std::string dataLayout;
  bool noOutput = false;
  bool outputAssembly = false;
  unsigned codeGenOptLevel = 0;
  bool disableOptimizationPasses = false;
  BCVersionNo outputBCVersion =
      BCVersionNo::DEFAULT; // 0 means the same as current upstream main.
  bool downgradeIR = false;

private:
  llvm::cl::OptionCategory cat{"Common command line options"};

  M::cl::MOpt<std::string, true> inputFilenameOp{
      cl::Positional, cl::desc("<input bitcode file>"),
      cl::value_desc("filename"), cl::location(inputFilename), cl::cat(cat)};

  M::cl::MOpt<std::string, true> outputFilenameOpt{
      "o", cl::desc("Override output filename"), cl::value_desc("filename"),
      cl::location(outputFilename), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO0Opt{
      "O0", cl::desc("Optimization level 0. Similar to mojo -O0. "),
      cl::location(optLevelO0), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO1Opt{
      "O1", cl::desc("Optimization level 1. Similar to mojo -O1. "),
      cl::location(optLevelO1), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO2Opt{
      "O2", cl::desc("Optimization level 2. Similar to mojo -O2. "),
      cl::location(optLevelO2), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO3Opt{
      "O3", cl::desc("Optimization level 3. Similar to mojo -O3. "),
      cl::location(optLevelO3), cl::cat(cat)};

  M::cl::MOpt<std::string, true> targetTripleOpt{
      "mtriple", cl::desc("Override target triple for module"),
      cl::location(targetTriple), cl::cat(cat)};

  M::cl::MOpt<std::string, true> dataLayoutOpt{
      "data-layout", cl::desc("data layout string to use"),
      cl::value_desc("layout-string"), cl::location(dataLayout), cl::cat(cat)};

  M::cl::MOpt<std::string, true> passPipelineOpt{
      "passes",
      cl::desc(
          "A textual description of the pass pipeline (same syntax as "
          "opt -passes=...). Mutually exclusive with -O0/-O1/-O2/-O3.\n"
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
      cl::value_desc("pipeline"), cl::location(passPipeline), cl::cat(cat)};

  M::cl::MOpt<bool, true> noOutputOpt{
      "disable-output", cl::desc("Do not write result bitcode file"),
      cl::Hidden, cl::location(noOutput), cl::cat(cat)};

  M::cl::MOpt<bool, true> outputAssemblyOpt{
      "S", cl::desc("Write output as LLVM assembly"),
      cl::location(outputAssembly), cl::cat(cat)};

  M::cl::MOpt<unsigned, true> codeGenOptLevelOpt{
      "codegen-opt-level",
      cl::desc("Override optimization level for codegen hooks, legacy PM only"),
      cl::location(codeGenOptLevel), cl::cat(cat)};

  M::cl::MOpt<bool, true> disableOptimizationPassesOpt{
      "disable-optimization-passes",
      cl::desc("Disable optimization passes and print input module. Useful to "
               "test Bitcode Writer"),
      cl::location(disableOptimizationPasses), cl::cat(cat)};

  M::cl::MOpt<BCVersionNo, true> irVersion{
      "output-bc-version",
      cl::desc("output bitcode llvm version"),
      cl::Hidden,
      llvm::cl::values(
          clEnumValN(BCVersionNo::DEFAULT, "default",
                     "Default bitcode version, no downgrading."),
          clEnumValN(BCVersionNo::LLVM17, "llvm17", "Bitcode version 17."),
          clEnumValN(BCVersionNo::LLVM19, "llvm19", "Bitcode version 19."),
          clEnumValN(BCVersionNo::LLVM21, "llvm21", "Bitcode version 21.")),
      cl::location(outputBCVersion),
      cl::init(BCVersionNo::DEFAULT),
      cl::cat(cat)};

  M::cl::MOpt<bool, true> downgradeIROpt{
      "downgrade-llvm-ir",
      cl::desc(
          "Run LLVMIRDowngrade pass for llvm backends with older versions."),
      cl::location(downgradeIR), cl::cat(cat)};
};

enum OutputKind {
  OK_NoOutput,
  OK_OutputAssembly,
  OK_OutputBitcode,
  OK_OutputThinLTOBitcode,
};
} // anonymous namespace

static CodeGenOptLevel getCodeGenOptLevel(const CLOptions &clOptions) {
  return static_cast<CodeGenOptLevel>(unsigned(clOptions.codeGenOptLevel));
}

static ModulePassManager
buildPipeline(PassBuilder &pb, const CLOptions &clOptions, Triple triple) {
  ModulePassManager mpm;

  // If an explicit pass pipeline is specified via -passes, use it directly.
  if (!clOptions.passPipeline.empty()) {
    if (auto err = pb.parsePassPipeline(mpm, clOptions.passPipeline)) {
      errs() << "error: failed to parse pass pipeline '"
             << clOptions.passPipeline << "': " << toString(std::move(err))
             << "\n";
      exit(1);
    }
    return mpm;
  }

  // Otherwise, build the full KGEN optimization pipeline for the given level.
  M::KGEN::CompilationOptions options(/*optimizationLevel=*/-1U);
  options.targetTriple = triple.str();
  if (clOptions.optLevelO0)
    options.optimizationLevel = 0;
  if (clOptions.optLevelO1)
    options.optimizationLevel = 1;
  if (clOptions.optLevelO2)
    options.optimizationLevel = 2;
  if (clOptions.optLevelO3)
    options.optimizationLevel = 3;

  if (options.optimizationLevel == -1U) {
    llvm_unreachable(
        "Specify an optimization level (-O0/-O1/-O2/-O3) or a custom pipeline "
        "(-passes=...).");
  }
  mpm = M::KGEN::buildLLVMOptimizationPipeline(pb, options);

  return mpm;
}

// Custom hack to handle Metal that is not supported in upstream
// Replace air64 with arm64
static std::string fixTargetTriple(std::string triple) {
  if (triple.find("air64") != std::string::npos)
    triple = triple.replace(triple.find("air64"), 5, "arm64");
  return triple;
}

int main(int argc, char **argv) {
  static codegen::RegisterCodeGenFlags cfg;
  CLOptions clOptions(argc, argv);
  InitLLVM llvm(argc, argv);

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry &registry = *PassRegistry::getPassRegistry();
  initializeCore(registry);
  initializeScalarOpts(registry);
  initializeVectorization(registry);
  initializeIPO(registry);
  initializeAnalysis(registry);
  initializeTransformUtils(registry);
  initializeInstCombine(registry);
  initializeTarget(registry);
  // For codegen passes, only passes that do IR to IR transformation are
  // supported.
  initializeExpandIRInstsLegacyPassPass(registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(registry);
  initializeSelectOptimizePass(registry);
  initializeInlineAsmPreparePass(registry);
  initializeCodeGenPrepareLegacyPassPass(registry);
  initializeAtomicExpandLegacyPass(registry);
  initializeWinEHPreparePass(registry);
  initializeDwarfEHPrepareLegacyPassPass(registry);
  initializeSafeStackLegacyPassPass(registry);
  initializeSjLjEHPreparePass(registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(registry);
  initializeGlobalMergePass(registry);
  initializeIndirectBrExpandLegacyPassPass(registry);
  initializeInterleavedLoadCombinePass(registry);
  initializeInterleavedAccessPass(registry);
  initializePostInlineEntryExitInstrumenterPass(registry);
  initializeUnreachableBlockElimLegacyPassPass(registry);
  initializeExpandReductionsPass(registry);
  initializeWasmEHPreparePass(registry);
  initializeWriteBitcodePassPass(registry);
  initializeReplaceWithVeclibLegacyPass(registry);
  initializeJMCInstrumenterPass(registry);
  initializeRuntimeLibraryInfoWrapperPass(registry);
  initializeLibcallLoweringInfoWrapperPass(registry);

  // Register the Target and CPU printer for --version.
  cl::AddExtraVersionPrinter(sys::printDefaultTargetAndDetectedCPU);

  cl::ParseCommandLineOptions(
      argc, argv, "llvm .bc -> .bc modular optimizer and analysis printer\n");

  errs() << "DEPRECATION WARNING: kgen-llvm-opt is deprecated and will be "
            "removed in the future. Use kgen-opt instead;\n";

  bool useCustomizedBitcodeWriter =
      (clOptions.outputBCVersion != BCVersionNo::DEFAULT);

  LLVMContext context;
  SMDiagnostic err;
  std::unique_ptr<Module> module;
  auto setDataLayout = [&](StringRef irTriple,
                           StringRef irLayout) -> std::optional<std::string> {
    if (!clOptions.dataLayout.empty())
      return std::nullopt;

    // If an explicit data layout is already defined in the IR, don't infer.
    if (!irLayout.empty())
      return std::nullopt;

    // If an explicit triple was specified (either in the IR or on the
    // command line), use that to infer the default data layout. However, the
    // command line target triple should override the IR file target triple.
    std::string tripleStr = clOptions.targetTriple.empty()
                                ? irTriple.str()
                                : Triple::normalize(clOptions.targetTriple);

    tripleStr = fixTargetTriple(tripleStr);

    // If the triple string is still empty, we don't fall back to
    // sys::getDefaultTargetTriple() since we do not want to have differing
    // behaviour dependent on the configured default triple. Therefore, if the
    // user did not pass -mtriple or define an explicit triple/datalayout in
    // the IR, we should default to an empty (default) DataLayout.
    if (tripleStr.empty())
      return std::nullopt;

    // Otherwise we infer the DataLayout from the target machine.
    Expected<std::unique_ptr<TargetMachine>> expectedTM =
        codegen::createTargetMachineForTriple(tripleStr,
                                              getCodeGenOptLevel(clOptions));
    if (!expectedTM) {
      errs() << argv[0] << ": warning: failed to infer data layout: "
             << toString(expectedTM.takeError()) << "\n";
      return std::nullopt;
    }
    return (*expectedTM)->createDataLayout().getStringRepresentation();
  };

  module = parseIRFile(clOptions.inputFilename, err, context,
                       ParserCallbacks(setDataLayout));

  if (!module) {
    err.print(argv[0], errs());
    return 1;
  }

  OutputKind outputKind = OK_NoOutput;
  if (!clOptions.noOutput)
    outputKind =
        clOptions.outputAssembly ? OK_OutputAssembly : OK_OutputBitcode;

  std::unique_ptr<ToolOutputFile> out;
  if (clOptions.noOutput) {
    if (!clOptions.outputFilename.empty())
      errs() << "WARNING: The -o (output filename) option is ignored when\n"
                "the --disable-output option is used.\n";
  } else {
    // Default to standard output.
    if (clOptions.outputFilename.empty())
      clOptions.outputFilename = "-";

    std::error_code errorCode;
    sys::fs::OpenFlags flags =
        clOptions.outputAssembly ? sys::fs::OF_TextWithCRLF : sys::fs::OF_None;
    out.reset(new ToolOutputFile(clOptions.outputFilename, errorCode, flags));
    if (errorCode) {
      errs() << errorCode.message() << '\n';
      return 1;
    }
  }

  if (!clOptions.targetTriple.empty())
    module->setTargetTriple(Triple(Triple::normalize(clOptions.targetTriple)));

  const bool isMetalTriple = M::KGEN::isMetalTriple(module->getTargetTriple());
  Triple moduleTriple(fixTargetTriple(module->getTargetTriple().str()));
  TargetLibraryInfoImpl tlii(moduleTriple);
  std::string cpuStr, featuresStr;
  std::unique_ptr<TargetMachine> targetMachine;

  if (isMetalTriple || moduleTriple.getArch()) {
    const TargetOptions options =
        codegen::InitTargetOptionsFromCodeGenFlags(moduleTriple);
    cpuStr = codegen::getCPUStr();
    featuresStr = codegen::getFeaturesStr();
    Expected<std::unique_ptr<TargetMachine>> expectedTM =
        codegen::createTargetMachineForTriple(moduleTriple.str(),
                                              getCodeGenOptLevel(clOptions));
    if (auto e = expectedTM.takeError()) {
      errs() << argv[0] << ": WARNING: failed to create target machine for '"
             << moduleTriple.str() << "': " << toString(std::move(e)) << "\n";
    } else {
      targetMachine = std::move(*expectedTM);
    }
  } else if (moduleTriple.getArchName() != "unknown" &&
             moduleTriple.getArchName() != "") {
    errs() << argv[0] << ": unrecognized architecture '"
           << moduleTriple.getArchName() << "' provided.\n";
    return 1;
  }

  // Override function attributes based on cpuStr, featuresStr, and command line
  // flags.
  codegen::setFunctionAttributes(*module, cpuStr, featuresStr);

  llvm::PassInstrumentationCallbacks pic;
  PassBuilder pb(targetMachine.get(), PipelineTuningOptions(),
                 /*PGOOpt=*/std::nullopt, &pic);
  M::KGEN::registerKGENLLVMPasses(pb);
  ModulePassManager mpm;
  if (!clOptions.disableOptimizationPasses)
    mpm = buildPipeline(pb, clOptions, module->getTargetTriple());
  if (clOptions.downgradeIR)
    M::KGEN::addLLVMIRDowngradePass(mpm);

  switch (outputKind) {
  case OK_NoOutput:
    break; // No output pass needed.
  case OK_OutputAssembly:
    mpm.addPass(PrintModulePass(out->os(), "",
                                /*ShouldPreserveAssemblyUseListOrder=*/false,
                                /*EmitSummaryIndex=*/false));
    break;
  case OK_OutputBitcode:
    // For metal use custom bitcode writer that emits AIR. That helps to test it
    // too.
    if (!isMetalTriple && !useCustomizedBitcodeWriter) {
      mpm.addPass(BitcodeWriterPass(out->os(),
                                    /*ShouldPreserveBitcodeUseListOrder=*/false,
                                    /*EmitSummaryIndex=*/false,
                                    /*EmitModuleHash=*/false));
    }
    break;
  case OK_OutputThinLTOBitcode:
    llvm_unreachable("Not implemented.");
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

  // Don't verify IR with upstream main if we downgrade it.
  if (!clOptions.downgradeIR && verifyModule(*module, &llvm::errs()))
    return 1;

  if (isMetalTriple && outputKind == OK_OutputBitcode) {
    M::KGEN::LLVM::WriteBitcode17ToFile(*module, out->os(),
                                        /*ShouldPreserveUseListOrder = */ false,
                                        /*ModuleSummaryIndex =*/nullptr,
                                        /*GenerateHash = */ false,
                                        /*ModuleHash = */ nullptr);
  } else if (outputKind == OK_OutputBitcode) {
    switch (clOptions.outputBCVersion) {
    case BCVersionNo::LLVM17:
      M::KGEN::LLVM::WriteBitcode17ToFile(
          *module, out->os(),
          /*ShouldPreserveUseListOrder = */ false,
          /*ModuleSummaryIndex =*/nullptr,
          /*GenerateHash = */ false,
          /*ModuleHash = */ nullptr);
      break;

    case BCVersionNo::LLVM19:
      M::KGEN::LLVM::WriteBitcode19ToFile(
          *module, out->os(),
          /*ShouldPreserveUseListOrder = */ false,
          /*ModuleSummaryIndex =*/nullptr,
          /*GenerateHash = */ false,
          /*ModuleHash = */ nullptr);
      break;
    case BCVersionNo::LLVM21:
      M::KGEN::LLVM::WriteBitcode21ToFile(
          *module, out->os(),
          /*ShouldPreserveUseListOrder = */ false,
          /*ModuleSummaryIndex =*/nullptr,
          /*GenerateHash = */ false,
          /*ModuleHash = */ nullptr);
      break;
    case BCVersionNo::DEFAULT:
      break;
    }
  }

  // Declare success.
  if (outputKind != OK_NoOutput)
    out->keep();
  return 0;
}
