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
// This file defines the options for developing the compiler that the `kgen`
// tools share.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_DEVELOPEROPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_DEVELOPEROPTIONS_TD

include "Common/Marshalling.td"
include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Developer Options
//===----------------------------------------------------------------------===//

def DeveloperOptionGroup : OptionGroup<"Developer options">, Index<25>;

defm mllvm : CommandOption<["-"], "mllvm", DeveloperOptionGroup,
  MetaVarName<"ARG">,
  HelpText<"Passes ARG to LLVM and MLIR's command-line options. May be given "
           "more than once.">>;

def time_trace : Flag<["--", "-"], "time-trace">,
  Group<DeveloperOptionGroup>,
  HelpText<"Writes a time profile to kgen.trace.json in the derived "
           "directory.">,
  MarshallingInfoFlag<DriverOpts<"timeTrace">>;

defm time_trace_granularity : CommandOption<["--", "-"],
  "time-trace-granularity", DeveloperOptionGroup,
  MetaVarName<"MICROSECONDS">,
  HelpText<"Sets the minimum duration of an event in the time profile.">>,
  MarshallingInfoInt<DriverOpts<"timeTraceGranularity">, "0", "int">;

#endif // KGEN_TOOLS_MOJO_COMMON_DEVELOPEROPTIONS_TD
