//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-demangle.h"
#include "../../common/Telemetry.h"

#include "AsyncRT/Init/Init.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/LLVMForwardDecls.h"

#include "mlir/IR/DialectRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

#define DRIVER_OPTIONS_PATH "Demangle/DemangleOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DemangleOptTable : public llvm::opt::PrecomputedOptTable {
  DemangleOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};
} // namespace

/// Given a user-provided string that could be a mangled symbol name, prints the
/// demangled name to stdout. Returns an integer representing a successful exit
/// code if demangling succeeded, otherwise returns a failure code.
static int demangle(const State &state) {
  // Parse command line arguments.
  DemangleOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Demangle/DemangleOptionsHelpText.inc"
    );
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  if (args.hasMultipleArgs(options::OPT_INPUT)) {
    std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
    return state.reportError(
        llvm::formatv("only one name can be demangled at a time; "
                      "cannot demangle both '{0}' and '{1}'",
                      inputs[0], inputs[1]));
  }

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext("mojo", Init::Options());
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Initialize telemetry.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), args);

  // Initialize the MLIR context with all of KGEN's dialects.
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  mlir::MLIRContext context{registry};

  // If no name was provided on the command-line, read one from stdin.
  std::string name =
      args.getLastArgValue(options::OPT_INPUT, /*Default=*/"").str();
  if (!args.hasArg(options::OPT_INPUT)) {
    SmallString<llvm::sys::fs::DefaultReadChunkSize> buffer;
    if (llvm::Error err = llvm::sys::fs::readNativeFileToEOF(
            llvm::sys::fs::getStdinHandle(), buffer))
      return state.reportError("cannot read name from stdin: " +
                               errorToErrorCode(std::move(err)).message());
    // Strip off any extra whitespace or control characters.
    name = buffer.str().trim();
  }

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  // Try to demangle the name and print it to stdout.
  FailureOr<MangledSymbol> mangled =
      MangledSymbol::demangle(StringAttr::get(&context, name));
  if (failed(mangled))
    return state.reportError("demangling failed");

  llvm::outs() << *mangled << "\n";
  return EXIT_SUCCESS;
}

void M::registerDemangleSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("demangle", demangle);
}
