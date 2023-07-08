//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>

using namespace M;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
M::openMojoInputFile(StringRef path) {
  if (!path.ends_with(".mojo") && !path.ends_with(".🔥"))
    return Error(llvm::formatv(
        "cannot open '{0}', since it does not appear to be a Mojo file "
        "(it does not end in '.mojo' or '.🔥')",
        path));

  // Open the input file, or exit with an error.
  std::string inputError;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      mlir::openInputFile(path, &inputError);
  if (!buffer)
    return Error(inputError);

  return std::move(buffer);
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

int State::reportError(Twine errorMessage) const {
  llvm::errs() << programName << ": error: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

int State::printHelp(bool plainText, Twine helpText) const {
  llvm::ErrorOr<std::string> man = llvm::sys::findProgramByName("man");
  if (!plainText && man) {
    // Eventually, Mojo driver man pages will be installed at locations
    // typically on the `man` search path, like `/usr/local/share/man`. However,
    // in addition to these, `man` also searches for locations relative to
    // paths in the user's PATH environment variable. So, for example, if a user
    // has `PATH=/foo/bin`, `man` will search `/foo/share/man`.
    //
    // Currently, Mojo developers are the only ones with access to the mojo
    // driver, and they all use start-modular shellscripts that inserts the
    // build directory into their PATH. As a result, `man` can find the driver
    // man pages relative to this directory, given only their name.
    //
    // We compute that name here: if it's a subcommand, append it to the program
    // name.
    std::string name = std::filesystem::path(programName).filename().string();
    if (subcommand)
      name = name + "-" + subcommand;

    // `man` should be able to find the man page successfully, but just in case
    // it can't (maybe the user's installation is messed up, or maybe they
    // deleted the man page file accidentally), we still wish to print help
    // text. Attempt to invoke `man` with its `stdin` and `stderr` disconnected,
    // and if it fails, fallthrough to other backup behavior.
    const std::optional<StringRef> redirects[] = {
        /*stdin*/ "",
        /*stdout*/ std::nullopt,
        /*stderr*/ "",
    };
    if (!llvm::sys::ExecuteAndWait(man.get(), {man.get(), "1", name},
                                   /*Env=*/std::nullopt, redirects))
      return EXIT_SUCCESS;
  }

  llvm::outs() << helpText;
  return EXIT_SUCCESS;
}

//===----------------------------------------------------------------------===//
// SubcommandRegistry
//===----------------------------------------------------------------------===//

void SubcommandRegistry::addCallback(StringRef subcommand,
                                     SubcommandRegistry::Callback callback) {
  std::string cmd = subcommand.str();
  assert(callbacks.count(cmd) == 0 && "subcommand already registered");
  assert(callback && "callback cannot be empty");
  callbacks.insert({cmd, callback});
}

ErrorOr<SubcommandRegistry::Callback>
SubcommandRegistry::getCallback(StringRef subcommand) {
  auto it = callbacks.find(subcommand.str());
  if (it != callbacks.end())
    return it->second;

  // The user provided a subcommand name we don't recognize; return an error
  // message.
  std::string message = ("no such command '" + subcommand + "'").str();

  // If there are any close matches, point those out in the message.
  std::string nearest;
  unsigned minDistance = std::numeric_limits<unsigned>::max();
  for (const auto &kv : callbacks) {
    unsigned distance = subcommand.edit_distance(kv.first());
    if (distance < minDistance) {
      minDistance = distance;
      nearest = kv.first();
    }
  }
  if (minDistance <= 2)
    message += ". Did you mean '" + nearest + "'?";

  return Error(message);
}
