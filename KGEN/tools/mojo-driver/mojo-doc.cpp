//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-doc.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/MojoParser.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <filesystem>

using namespace M;
using namespace M::KGEN;

namespace {
/// Options that apply only to the `doc` subcommand.
struct DocOptions {
  /// The `doc` subcommand itself.
  llvm::cl::SubCommand doc{
      "doc",
      "Translate source file doc strings into a structured output format."};

  /// The user-provided path to a Mojo source file. The doc strings in this file
  /// will be parsed and used to generate structured output representing those
  /// doc strings.
  cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                     cl::desc("<path to Mojo source file>"),
                                     llvm::cl::sub(doc)};

  /// The path to which output will be written.
  cl::opt<std::string> outputFilename{
      "o",
      cl::desc("The path to which an output file will be written. If not "
               "provided, output is written to stdout."),
      cl::value_desc("path"), cl::init("-"), llvm::cl::sub(doc)};

  /// Zero or more paths that are searched when the parser attempts to resolve
  /// a Mojo source file import.
  cl::list<std::string> includePaths{
      "I",
      cl::desc("Append the given path to the list of "
               "directories to search for included Mojo files."),
      cl::value_desc("path"), llvm::cl::sub(doc)};

  /// Whether to validate doc strings.
  cl::opt<bool> validate{
      "validate",
      cl::desc("Validate doc strings as they are parsed. When "
               "enabled, warning diagnostics are emitted as invalid "
               "doc strings are parsed."),
      llvm::cl::sub(doc)};
};
} // namespace

/// A global set of options for the `doc` subcommand. This must be instantiated
/// before parsing command-line arguments.
static llvm::ManagedStatic<DocOptions> options;

/// Given the path to a Mojo source file, opens and parses that file's doc
/// strings in order to generate structured output (currently JSON). Returns an
/// integer representing a successful exit code is documentation generation
/// succeeded, otherwise returns a failure code.
static int doc(const State &state) {
  // Reject input files that do not appear to be Mojo files (this includes stdin
  // "-").
  StringRef inputPath = options->inputFilename;
  if (!inputPath.ends_with(".mojo") && !inputPath.ends_with(".🔥"))
    return state.reportError("cannot open '" + inputPath +
                             "', since it does not appear to be a Mojo file "
                             "(it does not end in '.mojo' or '.🔥')");

  // Open the input file, or exit with an error.
  std::string inputError;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      mlir::openInputFile(inputPath, &inputError);
  if (!buffer)
    return state.reportError(inputError);

  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  // Collect only those include paths that actually refer to directories on the
  // host filesystem. (Mojo's parser searches the source manager's include
  // directories when resolving imports.)
  std::vector<std::string> includeDirs;
  includeDirs.reserve(options->includePaths.size());
  for (auto &path : options->includePaths)
    if (std::filesystem::is_directory(path))
      includeDirs.push_back(path);
  sourceManager.setIncludeDirs(includeDirs);

  // We don't allow users to configure LLCL runtime options, such as the
  // allocator or the work queue threading model.
  mlir::MLIRContext context;
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());
  MojoParserConfig parserConfig(&context, runtime, CompilationOptions());
  parserConfig.validateDocStrings = options->validate;

  // We also don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timingScope = timingManager.getRootScope();

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out =
      mlir::openOutputFile(options->outputFilename, &outputError);
  if (!out)
    return state.reportError(outputError);

  if (failed(
          generateMojoDoc(sourceManager, parserConfig, out->os(), timingScope)))
    return state.reportError("could not generate documentation");
  out->keep();
  return EXIT_SUCCESS;
}

void M::registerDocSubCommand(SubCommandRegistry &registry) {
  registry.addCallback(&options->doc, doc);
}
