//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Driver/DriverSupport.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/ErrorOr.h"

#include "Support/Filesystem/Paths.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
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

  std::error_code ec;
  std::filesystem::path fullPath = std::filesystem::absolute(path.str(), ec);
  if (ec) {
    return Error(
        llvm::formatv("failed to resolve the absolute path for '{0}': {1}",
                      path.str(), ec.message()));
  }

  // Open the input file, or exit with an error.
  std::string inputError;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      mlir::openInputFile(fullPath.string(), &inputError);
  if (!buffer)
    return Error(inputError);

  return std::move(buffer);
}

ErrorOr<std::filesystem::path>
M::resolveMojoInputFileOrPackage(StringRef path) {
  std::error_code ec;
  std::filesystem::path fullPath = std::filesystem::absolute(path.str(), ec);
  if (ec) {
    return Error(
        llvm::formatv("failed to resolve the absolute path for '{0}': {1}",
                      path.str(), ec.message()));
  }

  std::string ext = fullPath.extension().string();
  if (!llvm::is_contained({".mojo", ".🔥", ".mojopkg", ".📦"}, ext) &&
      !Filesystem::isMojoSourcePackagePath(fullPath)) {
    return Error(llvm::formatv(
        "cannot open '{0}', since it does not appear to be a Mojo file "
        "(it does not end in '.mojo', '.🔥', '.mojopkg', or '.📦') or a Mojo "
        "source package",
        path));
  }
  return fullPath;
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

int State::reportError(const Twine &message) const {
  switch (diagnosticFormat) {
  case DiagnosticFormat::Text:
    llvm::errs() << programName << ": error: " << message << '\n';
    break;
  case DiagnosticFormat::JSON:
    llvm::errs() << toJSON(json::Diagnostic{json::DiagnosticKind::Error,
                                            message.str()})
                 << '\n';
    break;
  }
  return EXIT_FAILURE;
}

void State::reportWarning(const Twine &message) const {
  switch (diagnosticFormat) {
  case DiagnosticFormat::Text:
    llvm::errs() << programName << ": warning: " << message << '\n';
    break;
  case DiagnosticFormat::JSON:
    llvm::errs() << toJSON(json::Diagnostic{json::DiagnosticKind::Warning,
                                            message.str()})
                 << '\n';
    break;
  }
}

int State::parseDiagnosticFormatArguments(
    llvm::opt::InputArgList &args,
    llvm::opt::OptSpecifier diagnosticFormatOptionID) {
  StringLiteral kDiagnosticFormatText = "text";
  StringLiteral kDiagnosticFormatJSON = "json";
  StringRef format =
      args.getLastArgValue(diagnosticFormatOptionID, kDiagnosticFormatText);
  if (!llvm::is_contained({kDiagnosticFormatText, kDiagnosticFormatJSON},
                          format)) {
    return reportError(
        llvm::formatv("invalid diagnostic format '{0}', expected one of: `{1}` "
                      "(the default), or `{2}`",
                      format, kDiagnosticFormatText, kDiagnosticFormatJSON));
  }

  diagnosticFormat = llvm::StringSwitch<DiagnosticFormat>(format)
                         .Case(kDiagnosticFormatText, DiagnosticFormat::Text)
                         .Case(kDiagnosticFormatJSON, DiagnosticFormat::JSON);
  return EXIT_SUCCESS;
}

#if __cplusplus >= 202002
int State::printHelp(std::u8string_view helpText) const {
  return printHelp(std::string_view(
      reinterpret_cast<const char *>(helpText.data()), helpText.size()));
}
#endif

int State::printHelp(std::string_view helpText) const {
  llvm::outs() << helpText;
  return EXIT_SUCCESS;
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

void State::assertNoUnusedArguments(const llvm::opt::InputArgList &args) const {
#ifndef NDEBUG
  int unused = 0;
  for (const llvm::opt::Arg *a : args)
    if (!a->isClaimed())
      unused = reportError("'" + a->getSpelling() + "' argument is unclaimed");
  assert(unused == 0 && "one or more arguments were unclaimed");
#endif
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
