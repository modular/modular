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
//
// This file is nearly equivalent to `Hi/hi.cpp`, you can read the comments
// there for an in-depth explanation of what's going on here.
//
//===----------------------------------------------------------------------===//

#include "bye.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>
#include <vector>

using namespace M;

#define DRIVER_OPTIONS_PATH "Bye/ByeOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct ByeOptTable : public llvm::opt::OptTable {
  ByeOptTable() : llvm::opt::OptTable(optionTables()) {}
};
} // namespace

static int bye(const State &subcommandState) {
  State state = subcommandState;
  ByeOptTable options;
  HelpTexts helpTexts(
#include "Bye/ByeOptionsHelpText.inc"
      ,
#include "Bye/ByeOptionsHelpHiddenText.inc"
  );
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode =
          state.parseArguments(options, helpTexts, args))
    return *exitCode;

  if (int result = state.rejectInvalidArguments(args))
    return result;

  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (!inputs.empty()) {
    for (std::string input : inputs)
      state.reportError("unrecognized argument '" + input + "'");
    return EXIT_FAILURE;
  }

  llvm::outs() << "Good-bye!\n";

  return EXIT_SUCCESS;
}

void M::registerByeSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("bye", bye);
}
