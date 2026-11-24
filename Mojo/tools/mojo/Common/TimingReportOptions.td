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
// This file defines the format and the destination of the timing and
// statistics reports.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_TIMINGREPORTOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_TIMINGREPORTOPTIONS_TD

include "Common/IncludeOptions.td"
include "Common/Marshalling.td"
include "Support/Driver/DriverCommand.td"

def timing_json: Flag<["--", "-"], "timing-json">,
  Group<CompilationOptionGroup>, Flags<[HelpHidden]>,
  HelpText<
    "Writes the reports that `--mlir-timing`, `--llvm-timing` and "
    "`--elaboration-stats` collect as one JSON object rather than as text. "
    "The object holds a `mlir` member, an `llvm` member and an `elaboration` "
    "member, and only the members the command asks for, so it is empty if "
    "the command asks for none. The `mlir` member keeps the tree or the list "
    "that `--mlir-timing-display` selects. The `llvm` and `elaboration` "
    "members hold one entry for each pipeline, with the name of the pipeline "
    "and the times of its passes, or its `comptime for` loops.">,
  MarshallingInfoFlag<DriverOpts<"timingJSON">>;

defm timing_file: CommandOptionHidden<["--", "-"], "timing-file",
  CompilationOptionGroup, MetaVarName<"FILE">,
  HelpText<
    "Writes the timing and elaboration reports to FILE rather than to "
    "stderr. The reports are text, or JSON with `--timing-json`.">>,
  MarshallingInfoString<DriverOpts<"timingFile">>;

#endif // KGEN_TOOLS_MOJO_COMMON_TIMINGREPORTOPTIONS_TD
