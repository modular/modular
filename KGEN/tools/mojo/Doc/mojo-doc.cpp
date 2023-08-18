//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-doc.h"
#include "../Common/Telemetry.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/MojoParser.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "KGEN/MojoParser/ASTDeclView.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/Driver/DriverSupport.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using namespace M::KGEN;

#define DRIVER_OPTIONS_PATH "Doc/DocOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DocOptTable : public llvm::opt::PrecomputedOptTable {
  DocOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Given the path to a Mojo source file, opens and parses that file's doc
/// strings in order to generate structured output (currently JSON). Returns an
/// integer representing a successful exit code is documentation generation
/// succeeded, otherwise returns a failure code.
static int doc(const State &state) {
  // Parse command line arguments.
  DocOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help, options::OPT_help_text)) {
    return state.printHelp(/*plainText=*/args.hasArg(options::OPT_help_text),
#include "Doc/DocOptionsHelpText.inc"
    );
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  if (!args.hasArg(options::OPT_INPUT))
    return state.reportError("no input file provided");
  if (args.hasMultipleArgs(options::OPT_INPUT)) {
    std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
    return state.reportError(llvm::formatv(
        "too many input files, cannot process both '{0}' and '{1}'", inputs[0],
        inputs[1]));
  }

  // We don't allow users to configure LLCL runtime options, such as the
  // allocator or the work queue threading model.
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());

  auto &telemetryCtx = runtime.emplaceContext<M::Telemetry::TelemetryContext>();

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  initializeTelemetry(telemetryCtx, state, args,
                      /*privateArgs=*/{options::OPT_I, options::OPT_o});

  // Open the input file, or exit with an error.
  auto bufferOrErr =
      openMojoInputFile(args.getLastArgValue(options::OPT_INPUT));
  if (bufferOrErr.isError())
    return state.reportError(bufferOrErr.getError());

  // Initialize the source manager with the input file buffer and all includes.
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*bufferOrErr), llvm::SMLoc());
  sourceManager.setIncludeDirs(args.getAllArgValues(options::OPT_I));

  mlir::MLIRContext context;
  CompilationOptions compilationOptions;
  MojoParserConfig parserConfig(&context, runtime, compilationOptions);
  parserConfig.warnMissingDocStrings =
      args.hasArg(options::OPT_warn_missing_dog_strings);

  // We also don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timingScope = timingManager.getRootScope();

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out = mlir::openOutputFile(
      args.getLastArgValue(options::OPT_o, "-"), &outputError);
  if (!out)
    return state.reportError(outputError);

  MojoParserContext parserContext(sourceManager, parserConfig);
  MojoASTDeclRef moduleDecl =
      parserContext.parseFile(sourceManager.getMainFileID());
  if (!moduleDecl)
    return state.reportError("could not generate documentation");

  std::unique_ptr<DeclView> declView = moduleDecl.getView();
  if (!declView)
    return state.reportError("could not generate documentation");

  llvm::json::OStream jsonOS(out->os(), /*IndentSize=*/2);
  jsonOS.value(declView->toJSON());
  out->keep();
  return EXIT_SUCCESS;
}

void M::registerDocSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("doc", doc);
}
