//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/ExecutionEngine/ExecutionEngine.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

namespace {
class CLOptions : public CLOptionsBase {

public:
  CLOptions(int argc, char **argv, bool skipInitLLVM = false)
      : CLOptionsBase(argc, argv, options, skipInitLLVM) {}

  OptionsBase options;
  std::string inputFilename{"-"};
  std::string outputPrefix{"-"};
  bool perFunctionSplit = false;

private:
  llvm::cl::OptionCategory CommonOptionsCategory{"Common command line options"};

  M::cl::MOpt<std::string, true> inputFileOpt{
      llvm::cl::Positional,
      llvm::cl::Required,
      llvm::cl::desc("Input filename"),
      llvm::cl::value_desc("filename"),
      llvm::cl::location(inputFilename),
      llvm::cl::cat(CommonOptionsCategory)};

  M::cl::MOpt<std::string, true> outputPrefixOpt{
      "output-prefix", llvm::cl::desc("output prefix"),
      llvm::cl::value_desc("output prefix"), llvm::cl::location(outputPrefix),
      llvm::cl::cat(CommonOptionsCategory)};

  M::cl::MOpt<bool, true> perFunctionSplitOpt{
      "per-func", llvm::cl::desc("split each function into separate modules"),
      llvm::cl::value_desc("split each function into separate modules"),
      llvm::cl::location(perFunctionSplit),
      llvm::cl::cat(CommonOptionsCategory)};
};

} // namespace

/// Reads a module from a file.  On error, messages are written to stderr
/// and null is returned.
static std::unique_ptr<Module> readModule(LLVMContext &Context,
                                          StringRef Name) {
  SMDiagnostic Diag;
  std::unique_ptr<Module> M = parseIRFile(Name, Diag, Context);
  if (!M)
    Diag.print("llvm-module-split", errs());
  return M;
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv, true);

  // Override the default version printer.
  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << "LLVM Module Split Tool:\n  ";
    os << "Modular version: " << version.major << '.' << version.minor << '.'
       << version.patch << version.label << "\n  ";
    os << "Git SHA: " << version.revision << "\n  ";
    os << "Build config: " << version.buildType << "\n\n";

    // Print the host target config.
    llvm::sys::printDefaultTargetAndDetectedCPU(os);
    // Print all registered targets.
    llvm::TargetRegistry::printRegisteredTargetsForVersion(os);
  });

  // Enable command line options for various MLIR internals.
  llvm::cl::ParseCommandLineOptions(argc, argv);

  LLVMModuleAndContext module;
  ErrorOrSuccess err = module.create(
      [&](LLVMContext &ctx) -> M::ErrorOr<std::unique_ptr<Module>> {
        if (std::unique_ptr<Module> module =
                readModule(ctx, clOptions.inputFilename))
          return module;
        return M::Error("could not load LLVM file");
      });
  if (err) {
    llvm::errs() << err.getError() << "\n";
    return -1;
  }

  std::unique_ptr<llvm::ToolOutputFile> output = nullptr;
  if (clOptions.outputPrefix == "-") {
    std::error_code error;
    output = std::make_unique<llvm::ToolOutputFile>(
        clOptions.outputPrefix, error, llvm::sys::fs::OF_None);
    if (error)
      exit(clOptions.options.reportError("Cannot open output file: '" +
                                         clOptions.outputPrefix +
                                         "':" + error.message()));
  }

  auto outputLambda = [&](LLVMModuleAndContext subModule,
                          std::optional<int64_t> idx) {
    if (clOptions.outputPrefix == "-") {
      output->os() << "##############################################\n";
      if (idx)
        output->os() << "# [LLVM Module Split: submodule " << *idx << "]\n";
      else
        output->os() << "# [LLVM Module Split: main module]\n";
      output->os() << "##############################################\n";
      output->os() << *subModule;
      output->os() << "\n";
    } else {
      std::string outPath;
      if (!idx)
        outPath = clOptions.outputPrefix + ".ll";
      else
        outPath = (clOptions.outputPrefix + "." + Twine(*idx) + ".ll").str();
      auto outFile = mlir::openOutputFile(outPath);
      if (!outFile)
        exit(clOptions.options.reportError("Cannot open output file: '" +
                                           outPath + "."));
      outFile->os() << *subModule;
      outFile->keep();
      llvm::outs() << "Write llvm module to " << outPath << "\n";
    }
  };

  if (clOptions.perFunctionSplit)
    splitPerFunction(std::move(module), outputLambda);
  else
    splitPerExported(std::move(module), outputLambda);

  if (output)
    output->keep();
  return 0;
}
