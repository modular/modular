//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-doc.h"

#include "Config/Version.h"
#include "Init/Init.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/MojoTooling/PublicASTDecl.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "MLRT/AsyncRT/Runtime/Allocator.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "MLRT/AsyncRT/Runtime/WorkQueue.h"
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
  DocOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
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
  } else if (args.hasArg(options::OPT_help_hidden)) {
    return state.printHelp(
#include "Doc/DocOptionsHelpHiddenText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format,
          /*disableWarningsId=*/llvm::opt::OptSpecifier(), options::OPT_werror,
          options::OPT_wno_error))
    return result;

  // Handle deprecated --validate-doc-strings flag as an alias for -Werror.
  // Only apply if user hasn't explicitly specified -Werror or -Wno-error.
  if (args.hasArg(options::OPT_validate_doc_strings) &&
      !args.hasArg(options::OPT_werror) &&
      !args.hasArg(options::OPT_wno_error)) {
    state.reportWarning(
        "--validate-doc-strings is deprecated, use -Werror instead");
    state.warningsAsErrors = true;
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

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext("mojo", Init::Options());
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

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
  MLIRContext context{registry};

  CompilationOptions compilationOptions;
  compilationOptions.warningsAsErrors = state.areWarningsAsErrors();
  LIT::ParserConfig parserConfig(&context, compilationOptions);
  parserConfig.diagnoseMissingDocStrings =
      args.hasArg(options::OPT_diagnose_missing_doc_strings);
  int maxNotes = 0;
  if (!args.getLastArgValue(options::OPT_max_notes).getAsInteger(10, maxNotes))
    parserConfig.maxNotesPerDiagnostic = maxNotes;
  parserConfig.stripFilePrefix =
      args.getLastArgValue(options::OPT_strip_file_prefix);
  parserConfig.docsBasePath = args.getLastArgValue(options::OPT_docs_base_path);

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

  std::unique_ptr<PublicDecl> publicDecl = moduleDecl.getDecl();
  if (!publicDecl)
    return state.reportError("could not generate documentation");

  llvm::json::OStream jsonOS(out->os(), /*IndentSize=*/2);

  const char *version = getModularVersionString();
  jsonOS.value(llvm::json::Object({
      {"decl", publicDecl->toJSON(parserContext)},
      {"version", llvm::formatv("0.{0}", version).str()},
  }));

  out->keep();

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  return EXIT_SUCCESS;
}

void M::registerDocSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("doc", doc);
}
