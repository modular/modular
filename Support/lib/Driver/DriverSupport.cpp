//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
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
  auto printPlainHelpText = [&]() {
    llvm::outs() << helpText;
    return EXIT_SUCCESS;
  };

  // If we're printing plain text, just do so now.
  if (plainText)
    return printPlainHelpText();

  // Otherwise, we'll attempt to display a manual page. If the `man` executable
  // isn't available, fall back to plain text.
  llvm::ErrorOr<std::string> man = llvm::sys::findProgramByName("man");
  if (!man)
    return printPlainHelpText();

  // We have a `man` executable, but can it display the manual page? A lot can
  // go wrong here:
  // - Aside from typical locations such as `/usr/local/share/man`, `man` also
  //   searches locations relative to the user's `PATH` to find manual pages.
  //   However, the user may not have their `PATH` configured correctly.
  // - Alternatively, the user may have deleted the manual file we're looking
  //   for, for some reason.
  // - The above scenarios return an unsuccessful exit code, but the following
  //   scenario returns a successful one: the user may be running this with a
  //   minimal Linux distribution or within a container that uses such a distro.
  //   In this case, `man` is stubbed out with a script that prints "This system
  //   has been minimized by removing packages and content that are not required
  //   on a system that users do not log into," and exits successfully.
  //
  // As a result, the only way for us to truly check whether `man` will succeed
  // is to actually run it and see if it contains the manual page title we're
  // looking for. If not, we fall back to plain text.
  std::string name = std::filesystem::path(programName).filename().string();
  if (subcommand)
    name = name + "-" + subcommand;

  // Create a temporary file to capture the output of our first `man`
  // invocation.
  std::error_code ec;
  std::filesystem::path tmpDirPath = std::filesystem::temp_directory_path(ec);
  if (ec)
    return printPlainHelpText();
  ErrorOr<TempFile> outOrErr =
      TempFile::create((tmpDirPath / "man-out-%%%%%%.txt").string());
  if (failed(outOrErr))
    return printPlainHelpText();
  std::string out = outOrErr->getPath().string();

  // Invoke `man`, directing its output to the file.
  const std::optional<StringRef> redirects[] = {
      /*stdin*/ "",
      /*stdout*/ out,
      /*stderr*/ "",
  };
  const StringRef args[] = {man.get(), "1", name};
  if (llvm::sys::ExecuteAndWait(man.get(), args,
                                /*Env=*/std::nullopt, redirects) != 0)
    // If the invocation itself failed, fall back to plain text.
    return printPlainHelpText();

  // Read the `man` output.
  auto bufferOrErr = llvm::MemoryBuffer::getFile(out);
  if (!bufferOrErr)
    return printPlainHelpText();
  // The moment of truth: all manual pages ought to contain the name of the
  // command, in uppercase, in the header. If the name we're looking for doesn't
  // appear in the output, we consider `man` to be unavailable, and fall back to
  // plain text.
  if (!bufferOrErr.get()->getBuffer().contains(StringRef(name).upper()))
    return printPlainHelpText();

  // At this point we're certain the `man` invocation will succeed, so do it
  // again without any redirects.
  return llvm::sys::ExecuteAndWait(man.get(), args);
}

int State::rejectUnknownArguments(
    llvm::opt::InputArgList &args,
    llvm::opt::OptSpecifier unknownOptionID) const {
  if (!args.hasArg(unknownOptionID))
    return EXIT_SUCCESS;

  for (llvm::opt::Arg *arg : args.filtered(unknownOptionID))
    reportError("unrecognized argument '" + arg->getSpelling() + "'");
  return EXIT_FAILURE;
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
