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
// This file defines the MLIR timing options, for the subcommands that time a
// parse or an MLIR pass pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_MLIRTIMINGOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_MLIRTIMINGOPTIONS_TD

include "Common/IncludeOptions.td"
include "Common/Marshalling.td"
include "Support/Driver/DriverCommand.td"

def mlir_timing: Flag<["--", "-"], "mlir-timing">,
  Group<CompilationOptionGroup>, Flags<[HelpHidden]>,
  HelpText<
    "Times each MLIR pass and analysis, printing a report to stderr when "
    "compilation finishes, which for `mojo run` is before the program starts. "
    "Passes that the compilation cache serves from an earlier build never "
    "run, so a warm cache reports the parse and nothing else; point "
    "`MODULAR_CACHE_DIR` at an empty directory to time a whole pipeline.">,
  MarshallingInfoFlag<DriverOpts<"mlirTiming">>;

defm mlir_timing_display: CommandOptionHidden<["--", "-"],
  "mlir-timing-display",
  CompilationOptionGroup, MetaVarName<"MODE">,
  HelpText<
    "How `--mlir-timing` groups its report. `tree` (the default) nests "
    "timings by pipeline structure; `list` aggregates per pass name, sorted "
    "by total time. The `list` mode merges the passes of all pipelines, so it "
    "cannot show the time of one offload target on its own.">>,
  MarshallingInfoString<DriverOpts<"mlirTimingDisplay">>;

#endif // KGEN_TOOLS_MOJO_COMMON_MLIRTIMINGOPTIONS_TD
