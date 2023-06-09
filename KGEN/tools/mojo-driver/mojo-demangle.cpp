//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-demangle.h"
#include "mojo-driver.h"

#include "KGEN/InitAllDialects.h"
#include "KGEN/LITDialect/LITOps.h"
#include "Support/CommandLine.h"
#include "Support/LLVMForwardDecls.h"

#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

namespace {
/// Options that apply only to the `demangle` subcommand.
struct DemangleOptions {
  /// The `demangle` subcommand itself.
  llvm::cl::SubCommand demangle{
      "demangle", "Demangle the name provided on the command line."};

  /// The user-provided name to demangle. This may be an empty string, as is the
  /// case when no arguments are provided. In that case, the name is read from
  /// stdin.
  cl::opt<std::string> name{llvm::cl::Positional, cl::desc("<name>"),
                            llvm::cl::sub(demangle)};
};
} // namespace

/// A global set of options for the `demangle` subcommand. This must be
/// instantiated before parsing command-line arguments.
static llvm::ManagedStatic<DemangleOptions> options;

/// Given a user-provided string that could be a mangled symbol name, prints the
/// demangled name to stdout. Returns an integer representing a successful exit
/// code if demangling succeeded, otherwise returns a failure code.
static int demangle(const State &state) {
  // Initialize the MLIR context with all of KGEN's dialects.
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  mlir::MLIRContext context(registry);

  // If no name was provided on the command-line, read one from stdin.
  std::string name = options->name;
  if (options->name.getNumOccurrences() == 0) {
    SmallString<llvm::sys::fs::DefaultReadChunkSize> buffer;
    if (llvm::Error err = llvm::sys::fs::readNativeFileToEOF(
            llvm::sys::fs::getStdinHandle(), buffer))
      return state.reportError("cannot read name from stdin: " +
                               errorToErrorCode(std::move(err)).message());
    // Strip off any extra whitespace or control characters.
    name = buffer.str().trim();
  }

  // Try to demangle the name and print it to stdout.
  FailureOr<MangledSymbol> mangled =
      MangledSymbol::demangle(StringAttr::get(&context, name));
  if (failed(mangled))
    return state.reportError("demangling failed");

  llvm::outs() << *mangled << "\n";
  return EXIT_SUCCESS;
}

void M::registerDemangleSubCommand(SubCommandRegistry &registry) {
  registry.addCallback(&options->demangle, demangle);
}
