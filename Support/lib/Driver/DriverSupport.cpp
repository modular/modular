//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Support/Driver/DriverSupport.h"
#include "Support/Driver/DiagnosticFormat.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"

#include "Support/Filesystem/Paths.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

using namespace M;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
M::openMojoInputFile(StringRef path) {
  if (path.ends_with("🔥")) {
    return Error(
        llvm::formatv("'{0}'. Extension no longer supported (`.🔥`).\n\n"
                      "A Mojo file glowed with a `.🔥`\n"
                      "But the devs said, \"In UTF-8 that is mired...\"\n"
                      "    They ditched the cute flame,\n"
                      "    Made the extension a name,\n"
                      "And now `.mojo` is all that's required.\n",
                      path));
  }
  if (!path.ends_with(".mojo"))
    return Error(llvm::formatv(
        "cannot open '{0}', since it does not appear to be a Mojo file "
        "(it does not end in '.mojo')",
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
  if (!llvm::is_contained({".mojo", ".mojoc"}, ext) &&
      !Filesystem::isMojoSourcePackagePath(fullPath)) {
    return Error(llvm::formatv(
        "cannot open '{0}', since it does not appear to be a Mojo file "
        "(it does not end in '.mojo' or '.mojoc') or a Mojo "
        "source package",
        path));
  }
  return fullPath;
}

static bool isInputArg(const llvm::opt::Arg *arg) {
  return arg->getOption().getKind() == llvm::opt::Option::InputClass;
}

std::vector<std::string> M::getInputValues(const llvm::opt::ArgList &args) {
  std::vector<std::string> values;
  for (const llvm::opt::Arg *arg : args) {
    if (isInputArg(arg)) {
      arg->claim();
      values.push_back(arg->getValue());
    }
  }
  return values;
}

const llvm::opt::Arg *M::getFirstInputArg(const llvm::opt::ArgList &args) {
  for (const llvm::opt::Arg *arg : args) {
    if (isInputArg(arg)) {
      arg->claim();
      return arg;
    }
  }
  return nullptr;
}

#if __cplusplus >= 202002
HelpTexts::HelpTexts(std::u8string_view help, std::u8string_view helpHidden)
    : HelpTexts(
          std::string_view(reinterpret_cast<const char *>(help.data()),
                           help.size()),
          std::string_view(reinterpret_cast<const char *>(helpHidden.data()),
                           helpHidden.size())) {}
#endif

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

int State::reportError(const Twine &message) const {
  switch (diagnosticFormat) {
  case DiagnosticFormat::Text:
    llvm::errs() << programName << ": error: " << message << '\n';
    break;
  case DiagnosticFormat::JSON:
    llvm::outs() << toJSON(json::Diagnostic{json::DiagnosticKind::Error,
                                            message.str()})
                 << '\n';
    break;
  }
  return EXIT_FAILURE;
}

void State::reportWarning(const Twine &message) const {
  if (disableWarnings)
    return;

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

std::optional<int>
State::parseArguments(const llvm::opt::OptTable &optTable,
                      const HelpTexts &helpTexts, llvm::opt::InputArgList &args,
                      bool stopAtFirstInput,
                      llvm::function_ref<int(std::string_view)> printHelpText) {
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  args = optTable.ParseArgs(arguments, missingIndex, missingCount);
  if (stopAtFirstInput) {
    auto input = llvm::find_if(args, isInputArg);
    if (input != args.end())
      args = optTable.ParseArgs(arguments.slice(0, (*input)->getIndex() + 1),
                                missingIndex, missingCount);
  }
  missingValueOption = missingCount ? args.getArgString(missingIndex) : nullptr;

  // The help options come from CommonOptions.td, which every command includes,
  // but each table gives them its own IDs.
  bool help = false;
  bool helpHidden = false;
  for (const llvm::opt::Arg *arg : args) {
    StringRef name = arg->getOption().getName();
    help |= name == "help";
    helpHidden |= name == "help-hidden";
  }
  if (!help && !helpHidden)
    return std::nullopt;
  std::string_view text = help ? helpTexts.help : helpTexts.helpHidden;
  return printHelpText ? printHelpText(text) : printHelp(text);
}

int State::rejectInvalidArguments(const llvm::opt::InputArgList &args,
                                  ArrayRef<std::string> errors) const {
  if (!errors.empty())
    return reportError(errors.front());
  int result = EXIT_SUCCESS;
  if (missingValueOption)
    result =
        reportError(Twine("missing value for '") + missingValueOption + "'");
  for (const llvm::opt::Arg *arg : args)
    if (arg->getOption().getKind() == llvm::opt::Option::UnknownClass)
      result =
          reportError("unrecognized argument '" + arg->getSpelling() + "'");
  return result;
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
                                     SubcommandRegistry::Callback callback,
                                     std::optional<StringRef> deprecatedFor) {
  std::string cmd = subcommand.str();
  assert(callbacks.count(cmd) == 0 && "subcommand already registered");
  assert(callback && "callback cannot be empty");
  callbacks.insert({cmd, {callback, deprecatedFor}});
}

ErrorOr<SubcommandRegistry::Callback>
SubcommandRegistry::getCallback(StringRef subcommand) {
  auto it = callbacks.find(subcommand.str());
  if (it != callbacks.end()) {
    if (std::optional<StringRef> deprecatedFor = it->second.second) {
      llvm::errs() << "warning: '" << subcommand << "' is deprecated";
      if (!deprecatedFor->empty())
        llvm::errs() << "; use '" << *deprecatedFor << "' instead\n";
    }
    return it->second.first;
  }

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
