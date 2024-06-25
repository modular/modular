//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-doc.h"
#include "../../common/Telemetry.h"

#include "Config/Version.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "LLCL/Init/Init.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/Driver/DiagnosticFormat.h"
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
static int doc(const State &subcommandState) {
  // Parse command line arguments.
  State state = subcommandState;
  DocOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Doc/DocOptionsHelpText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format))
    return result;
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

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext("mojo", Init::Options());
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), args,
      /*privateArgs=*/{options::OPT_I, options::OPT_o});

  // Resolve the input, or exit with an error.
  auto pathOrErr =
      resolveMojoInputFileOrPackage(args.getLastArgValue(options::OPT_INPUT));
  if (pathOrErr)
    return state.reportError(pathOrErr.getError());

  // Initialize the source manager with the appropriate diagnostic handler and
  // include paths.
  llvm::SourceMgr sourceManager;
  sourceManager.setDiagHandler(getDiagHandler(state.diagnosticFormat));
  sourceManager.setIncludeDirs(args.getAllArgValues(options::OPT_I));

  DialectRegistry registry;
  registerAllKGENDialects(registry);
  mlir::MLIRContext context{registry};

  CompilationOptions compilationOptions;
  LIT::ParserConfig parserConfig(&context, compilationOptions);
  parserConfig.diagnoseMissingDocStrings =
      args.hasArg(options::OPT_diagnose_missing_doc_strings);
  parserConfig.errorOnInvalidDocStrings =
      args.hasArg(options::OPT_validate_doc_strings);

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
  MojoASTDeclRef moduleDecl = parserContext.parseFileOrPackage(*pathOrErr);
  if (!moduleDecl || parserContext.wasErrorEmitted())
    return state.reportError("could not generate documentation");

  std::unique_ptr<DeclView> declView = moduleDecl.getView();
  if (!declView)
    return state.reportError("could not generate documentation");

  llvm::json::OStream jsonOS(out->os(), /*IndentSize=*/2);

  ModularVersion version = getModularVersion();
  jsonOS.value(llvm::json::Object({
      {"decl", declView->toJSON(parserContext)},
      {"version", llvm::formatv("{0}.{1}.{2}{3}", version.major, version.minor,
                                version.patch, version.label)
                      .str()},
  }));

  out->keep();

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  return EXIT_SUCCESS;
}

void M::registerDocSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("doc", doc);
}
