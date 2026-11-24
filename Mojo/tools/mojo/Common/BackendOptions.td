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
// This file defines the compilation options that only take effect when a tool
// compiles Mojo programs through to machine code and links them: LLVM pass
// timing, linking and bitcode libraries. Tools that stop at the parser don't
// include it, so they reject these options instead of ignoring them.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_BACKENDOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_BACKENDOPTIONS_TD

include "Common/CompilationOptions.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

// `mojo precompile` shares the MLIR timing and report options. They are
// included around `--llvm-timing` to keep the order of the help text.
include "Common/MLIRTimingOptions.td"

def llvm_timing: Flag<["--", "-"], "llvm-timing">,
  Group<CompilationOptionGroup>, Flags<[HelpHidden]>,
  HelpText<
    "Measures the time of each LLVM pass and analysis. Writes the LLVM report "
    "to stderr when the compilation is complete. The timers of LLVM are "
    "global to the process and are not safe for more than one thread. "
    "Therefore this option makes the compiler use one thread, and it "
    "overrides `--num-threads`. The report then shows the time that LLVM "
    "uses, but it does not show the cost of a parallel build. LLVM does not "
    "get the object code that the compilation cache keeps from an earlier "
    "build. Therefore the report is empty if the cache has the object code. "
    "Set `MODULAR_CACHE_DIR` to an empty directory to measure a full "
    "pipeline. The report holds the passes of the host and the passes of "
    "each `compile_offload` target together.">,
  MarshallingInfoFlag<DriverOpts<"llvmTiming">>;

include "Common/TimingReportOptions.td"

//===----------------------------------------------------------------------===//
// Linker Options
//===----------------------------------------------------------------------===//

def LinkOptionGroup : OptionGroup<"Linker options">, Index<15>;

defm Xlinker : CommandOption<["-"], "Xlinker",
  LinkOptionGroup, MetaVarName<"ARG">,
  HelpText<"Pass ARG to the linker.">>;

defm lld_path : CommandOption<["--", "-"], "lld-path",
  LinkOptionGroup, MetaVarName<"PATH">,
  HelpText<"Overrides the path to the `lld` linker used when linking. Takes "
           "precedence over the `MODULAR_MOJO_MAX_LLD_PATH` environment "
           "variable and the `mojo-max.lld_path` configuration value.">>,
  MarshallingInfoString<DriverOpts<"lldPath">>;

include "Common/IncrParsingOptions.td"
include "Common/BitcodeOptions.td"

#endif // KGEN_TOOLS_MOJO_COMMON_BACKENDOPTIONS_TD
