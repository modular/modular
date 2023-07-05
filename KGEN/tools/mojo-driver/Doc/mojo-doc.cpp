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
#include "Support/Driver/DriverSupport.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
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

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Doc/DocOptionsHelpText.inc"
    );
  }

  if (args.hasArg(options::OPT_UNKNOWN)) {
    int result = 1;
    for (llvm::opt::Arg *arg : args.filtered(options::OPT_UNKNOWN))
      result = state.reportError("unrecognized argument '" +
                                 arg->getSpelling() + "'\n");
    return result;
  }

  if (!args.hasArg(options::OPT_INPUT))
    return state.reportError("no input file provided");
  if (args.hasMultipleArgs(options::OPT_INPUT)) {
    std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
    return state.reportError(llvm::formatv(
        "too many input files, cannot process both '{0}' and '{1}'", inputs[0],
        inputs[1]));
  }

  // Reject input files that do not appear to be Mojo files (this includes stdin
  // "-").
  StringRef inputPath = args.getLastArgValue(options::OPT_INPUT);
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
  includeDirs.reserve(args.getAllArgValues(options::OPT_I).size());
  for (auto &path : args.getAllArgValues(options::OPT_I))
    if (std::filesystem::is_directory(path))
      includeDirs.push_back(path);
  sourceManager.setIncludeDirs(includeDirs);

  // We don't allow users to configure LLCL runtime options, such as the
  // allocator or the work queue threading model.
  mlir::MLIRContext context;
  LLCL::Runtime runtime(LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue());
  CompilationOptions compilationOptions;
  MojoParserConfig parserConfig(&context, runtime, compilationOptions);
  parserConfig.validateDocStrings = args.hasArg(options::OPT_validate);

  // We also don't allow users to configure the time profiler.
  mlir::DefaultTimingManager timingManager;
  mlir::TimingScope timingScope = timingManager.getRootScope();

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out = mlir::openOutputFile(
      args.getLastArgValue(options::OPT_o, "-"), &outputError);
  if (!out)
    return state.reportError(outputError);

  if (failed(
          generateMojoDoc(sourceManager, parserConfig, out->os(), timingScope)))
    return state.reportError("could not generate documentation");
  out->keep();
  return EXIT_SUCCESS;
}

void M::registerDocSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("doc", doc);
}
