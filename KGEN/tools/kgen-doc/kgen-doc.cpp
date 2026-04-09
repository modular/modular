//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Standalone doc-generation tool. Produces the same JSON output as `mojo doc`
// but avoids the LLDB and JIT dependencies that make the full `mojo` binary
// slow to build, enabling faster iteration on doc-gen tests.
//
//===----------------------------------------------------------------------===//

#include "Init/Init.h"
#include "KGEN/MojoTooling/DocGen.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Driver/DriverSupport.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;
using namespace llvm;

int main(int argc, char *argv[]) {
  llvm::InitLLVM x(argc, argv);

  //===--------------------------------------------------------------------===//
  // Command-line options (same names as `mojo doc` for test compatibility)
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input.mojo>"),
                                 cl::Required);

  cl::opt<std::string> outputFile(
      "o",
      cl::desc("Sets the path and filename for the JSON output. "
               "If not provided, output is written to stdout."),
      cl::value_desc("PATH"), cl::init("-"));

  cl::list<std::string> includePaths(
      "I",
      cl::desc("Appends the given path to the list of directories that "
               "kgen-doc will search for package/module dependencies."),
      cl::value_desc("PATH"));

  cl::opt<bool> diagnoseMissingDocStrings(
      "diagnose-missing-doc-strings",
      cl::desc("Emits diagnostic warnings for missing or partial doc strings."),
      cl::init(false));

  cl::opt<std::string> docsBasePath(
      "docs-base-path",
      cl::desc("Sets the path prefix for generated documentation links."),
      cl::value_desc("PATH"), cl::init(""));

  cl::opt<unsigned> maxNotesPerDiagnostic(
      "max-notes-per-diagnostic",
      cl::desc("Sets the maximum number of notes printed with a diagnostic."),
      cl::value_desc("INTEGER"), cl::init(10));

  cl::opt<std::string> stripFilePrefix(
      "strip-file-prefix",
      cl::desc("Strip this prefix from filenames used for diagnostics."),
      cl::value_desc("PATH"), cl::init(""), cl::Hidden);

  // Register -Werror, -Wno-error, and --validate-doc-strings so that cl:: does
  // not treat them as unknown options. The actual last-one-wins ordering logic
  // is handled by scanning argv below, before cl::ParseCommandLineOptions runs.
  cl::opt<bool> werrorOpt("Werror", cl::desc("Treat warnings as errors."),
                          cl::init(false), cl::Hidden);
  cl::opt<bool> wnoErrorOpt("Wno-error",
                            cl::desc("Do not treat warnings as errors."),
                            cl::init(false), cl::Hidden);
  cl::opt<bool> validateDocStringsOpt(
      "validate-doc-strings",
      cl::desc("Deprecated: use -Werror instead. "
               "Treat doc string warnings as errors."),
      cl::init(false), cl::Hidden);

  cl::opt<DiagnosticFormat> diagnosticFormat(
      "diagnostic-format",
      cl::desc("The format in which diagnostics are printed."),
      cl::values(clEnumValN(DiagnosticFormat::Text, "text",
                            "Print diagnostics as plain text (default)"),
                 clEnumValN(DiagnosticFormat::JSON, "json",
                            "Print diagnostics as JSON Lines")),
      cl::init(DiagnosticFormat::Text));

  cl::ParseCommandLineOptions(argc, argv,
                              "kgen-doc: Mojo documentation generator\n");

  //===--------------------------------------------------------------------===//
  // Determine -Werror/-Wno-error with last-one-wins semantics by scanning
  // argv in order (cl:: does not preserve flag ordering across two separate
  // opts).
  //
  // --validate-doc-strings is a deprecated alias for -Werror that only applies
  // when neither -Werror nor -Wno-error appears anywhere on the command line.
  //
  // TODO: This manual argv scan duplicates logic from mojo doc's
  // State::parseDiagnosticFormatArguments. If kgen-doc ever grows a shared
  // driver layer, unify the flag-resolution logic there.
  //===--------------------------------------------------------------------===//

  bool hasExplicitWerror = false;
  bool hasExplicitWnoError = false;
  bool hasValidateDocStrings = false;
  for (int i = 1; i < argc; ++i) {
    StringRef arg = argv[i];
    if (arg == "-Werror" || arg == "--Werror")
      hasExplicitWerror = true;
    else if (arg == "-Wno-error" || arg == "--Wno-error")
      hasExplicitWnoError = true;
    else if (arg == "--validate-doc-strings" || arg == "-validate-doc-strings")
      hasValidateDocStrings = true;
  }

  bool warningsAsErrors = false;
  if (hasExplicitWerror || hasExplicitWnoError) {
    // -Werror and -Wno-error present: last one wins.
    for (int i = 1; i < argc; ++i) {
      StringRef arg = argv[i];
      if (arg == "-Werror" || arg == "--Werror")
        warningsAsErrors = true;
      else if (arg == "-Wno-error" || arg == "--Wno-error")
        warningsAsErrors = false;
    }
  } else if (hasValidateDocStrings) {
    // --validate-doc-strings with no explicit -Werror/-Wno-error: deprecated
    // alias for -Werror.
    llvm::errs() << "kgen-doc: warning: --validate-doc-strings is deprecated, "
                    "use -Werror instead\n";
    warningsAsErrors = true;
  }

  //===--------------------------------------------------------------------===//
  // Create context
  //===--------------------------------------------------------------------===//

  M::ErrorOr<ContextRef> ctxOr =
      Init::createContext("kgen-doc", Init::Options(), "doc");
  if (ctxOr.isError()) {
    llvm::errs() << "kgen-doc: error: " << ctxOr.getError() << "\n";
    return EXIT_FAILURE;
  }
  // Keep ctx alive for the duration of the pipeline; it holds init/runtime
  // state that the parser depends on.
  ContextRef ctx = std::move(*ctxOr);

  //===--------------------------------------------------------------------===//
  // Resolve the input path
  //===--------------------------------------------------------------------===//

  M::ErrorOr<std::filesystem::path> pathOrErr =
      resolveMojoInputFileOrPackage(inputFile);
  if (pathOrErr) {
    llvm::errs() << "kgen-doc: error: " << pathOrErr.getError() << "\n";
    return EXIT_FAILURE;
  }

  //===--------------------------------------------------------------------===//
  // Set up MLIR context
  //===--------------------------------------------------------------------===//

  mlir::DialectRegistry registry;
  registerAllKGENDialects(registry);
  mlir::MLIRContext context{registry};

  //===--------------------------------------------------------------------===//
  // Open output file
  //===--------------------------------------------------------------------===//

  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out =
      mlir::openOutputFile(outputFile, &outputError);
  if (!out) {
    llvm::errs() << "kgen-doc: error: " << outputError << "\n";
    return EXIT_FAILURE;
  }

  //===--------------------------------------------------------------------===//
  // Build config and generate documentation
  //
  // Note: unlike `mojo doc`, kgen-doc intentionally omits
  // mlir::DefaultTimingManager / mlir::TimingScope. The timing scope has no
  // effect on parser behavior and is excluded here to keep the binary minimal.
  //===--------------------------------------------------------------------===//

  DocGenConfig config;
  config.warningsAsErrors = warningsAsErrors;
  config.diagnoseMissingDocStrings = diagnoseMissingDocStrings;
  config.maxNotesPerDiagnostic = maxNotesPerDiagnostic;
  config.stripFilePrefix = stripFilePrefix;
  config.docsBasePath = docsBasePath;
  config.includePaths = std::vector<std::string>(includePaths);
  config.diagnosticFormat = diagnosticFormat;

  if (!generateMojoDocJSON(*pathOrErr, context, config, out->os())) {
    llvm::errs() << "kgen-doc: error: could not generate documentation\n";
    return EXIT_FAILURE;
  }

  out->keep();
  return EXIT_SUCCESS;
}
