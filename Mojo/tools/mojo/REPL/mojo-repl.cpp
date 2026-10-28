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

#include "mojo-repl.h"
#include "../Common/LLDB.h"
#include "Init/Init.h"
#include "llvm/Option/ArgList.h"

using namespace M;

#define DRIVER_OPTIONS_PATH "REPL/REPLOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct REPLOptTable : public llvm::opt::OptTable {
  REPLOptTable() : llvm::opt::OptTable(optionTables()) {}
};
} // namespace

/// Launches the Mojo REPL, which is in fact an invocation of
/// `lldb --repl-language mojo`. Exits unsuccessfully if LLDB could not be found
/// in the user's PATH.
static int repl(const State &subcommandState) {
  State state = subcommandState;

  // Create our context.
  ErrorOr<ContextRef> ctxOr =
      Init::createContext("mojo", Init::Options(), "repl");
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Parse command line arguments. We forward most arguments to the underlying
  // invocation of lldb, and so don't check for invalid options.
  REPLOptTable options;
  HelpTexts helpTexts(
#include "REPL/REPLOptionsHelpText.inc"
      ,
#include "REPL/REPLOptionsHelpHiddenText.inc"
  );
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode =
          state.parseArguments(options, helpTexts, args))
    return *exitCode;

  SmallVector<std::string> lldbArgs = {"--one-line-before-file",
                                       "settings set show-progress false",
                                       "--repl-language", "mojo", "--repl"};
  llvm::append_range(lldbArgs, state.arguments);
  return invokeLLDB(state, lldbArgs);
}

void M::registerREPLSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("repl", repl);
}
