//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tool that primes the compilation cache of .mojopkgs.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/Package/Package.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"

using namespace M;

namespace {
/// A class to hold the common command-line options for the tool.
template <typename T>
struct CLOptions : public LLCL::RuntimeCLOptions, CommonCLOptions {
  CLOptions(int argc, char **argv, T &opts, bool skipInitLLVM = false)
      : LLCL::RuntimeCLOptions(opts),
        CommonCLOptions(argc, argv, opts, skipInitLLVM) {}

  SmallVector<std::string> includePaths;
};

/// The main options entry point.
class Options : public LLCL::RuntimeOptions, public CommonOptions {
public:
  CLOptions<Options> parser;

  Options(int argc, char **argv, bool skipInitLLVM = false)
      : parser(argc, argv, *this, skipInitLLVM) {}

  /// Get the include directories that exist on the file system.
  std::vector<std::string> getIncludePaths() const {
    std::vector<std::string> result;
    result.reserve(parser.includePaths.size());
    for (auto &path : parser.includePaths)
      if (std::filesystem::is_directory(path))
        result.push_back(path);
    return result;
  }

  M::cl::MListOpt<std::string, SmallVector<std::string>> includePaths{
      "I", cl::desc("Path to use to search for included files."),
      llvm::cl::location(parser.includePaths)};

  M::cl::MOpt<bool> disableCodegenPriming{
      "disable-codegen-priming",
      cl::desc("Disable priming the cache for the codegen archive."),
      llvm::cl::init(false)};
};
} // namespace

/// Prime the cache for the given package and compilation environment.
static ErrorOrSuccess
primeCacheForPackage(KGEN::LIT::PackageOp packageOp, TargetInfoAttr targetInfo,
                     LLCL::Runtime &runtime,
                     const KGEN::CompilationOptions &options,
                     bool disableCodegenPriming) {
  // Build a package link that we'll use to call into the compilation methods.
  OpBuilder builder(packageOp.getContext());
  OwningOpRef<KGEN::PackageLinkOp> packageLink =
      builder.create<KGEN::PackageLinkOp>(
          packageOp->getLoc(), packageOp.getNameAttr(),
          packageOp.getPostParseModuleAttr(),
          /*preElaborationModule=*/DenseResourceElementsAttr(),
          packageOp.getCompiledEnvAttr(),
          /*archives=*/ArrayRef<KGEN::PackageArchiveAttr>());

  // First specialize the module up to the pre-elaboration phase.
  ErrorOr<DenseResourceElementsAttr> preElabOr =
      specializePackageLinkForPreElaborationLinking(*packageLink, runtime,
                                                    options);
  if (preElabOr.isError())
    return preElabOr.takeError();
  if (disableCodegenPriming)
    return success();

  // Next, compile for the current target.
  ErrorOr<KGEN::PackageArchiveAttr> archiveOr = KGEN::loadAndElaborateBytecode(
      *packageLink, targetInfo, options, runtime);
  if (archiveOr.isError())
    return archiveOr.takeError();
  return success();
}

/// Execute the tool pipeline to prime the different caches for the input
/// package.
static LogicalResult runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                                     Options &clOptions) {
  // Read the input package.
  mlir::ParserConfig parserConfig(ctx, /*verifyAfterParse=*/false);
  OwningOpRef<KGEN::LIT::PackageOp> packageOp =
      readOpFromBytecodeFile<KGEN::LIT::PackageOp>(
          mgr.getMemoryBuffer(mgr.getMainFileID())->getMemBufferRef(),
          parserConfig);
  if (!packageOp)
    return failure(clOptions.reportError("failed to read input package"));
  KGEN::CompilationOptions options;

  // Get the target info for the current compilation environment.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, options.targetTriple, options.targetCpu, options.targetFeatures);
  if (targetInfoOr.isError())
    return targetInfoOr.takeError();
  std::unique_ptr<LLCL::Runtime> runtime = clOptions.createRuntime();
  std::vector<AnyAsyncValueRef> asyncValues;

  // Helper functor used to enqueue priming the cache for a specific
  // compilation.
  auto addCompilation = [&](KGEN::CompilationOptions options) {
    auto out = AsyncValueRef<Chain>::allocate(*runtime);
    LLCL::addTask(*runtime, [&, out = out.copy(),
                             options = std::move(options)]() mutable {
      ErrorOrSuccess resultOr =
          primeCacheForPackage(*packageOp, *targetInfoOr, *runtime, options,
                               clOptions.disableCodegenPriming);
      if (resultOr.isError())
        std::move(out).setToError(
            LLCL::getMLIRDiagnostic(resultOr.takeError(), packageOp->getLoc()));
      else
        std::move(out).emplace();
    });
    asyncValues.push_back(out.copy());
  };

  // Prime a "release" build.
  addCompilation(options);
  // Prime a "debug" build.
  options.optimizationLevel = 0;
  options.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;
  addCompilation(options);

  // Wait for all compilations to complete.
  LLCL::await(asyncValues);
  for (auto &asyncValue : asyncValues) {
    if (asyncValue.isError()) {
      return failure(
          clOptions.reportError(asyncValue.getDiagnostic().getMessage().get()));
    }
  }
  return success();
}

int main(int argc, char **argv) {
  Options clOptions(argc, argv);

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Override the default version printer.
  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << "KGEN compiler:\n  ";
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
  mlir::registerMLIRContextCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Make sure the input is actually a package.
  StringRef inputFileName(clOptions.inputFilename);
  if (!inputFileName.ends_with(".mojopkg") && !inputFileName.ends_with(".📦"))
    return clOptions.reportError("input file must be a Mojo package");

  // Set up the input file(s).
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(clOptions.getIncludePaths());
  sourceManager.AddNewSourceBuffer(clOptions.openInputFileOrExit(),
                                   llvm::SMLoc());

  return failed(clOptions.configureMLIRContextAndExecute(
      sourceManager, [&](MLIRContext *ctx) {
        DialectRegistry registry;
        registerAllKGENDialects(registry);
        registerKGENToLLVMTranslation(registry);
        ctx->appendDialectRegistry(registry);
        ctx->loadAllAvailableDialects();

        return runToolPipeline(ctx, sourceManager, clOptions);
      }));
}
