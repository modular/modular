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
// compiles Mojo programs through to machine code and links them: pass timing,
// linking and bitcode libraries. Tools that stop at the parser don't include
// it, so they reject these options instead of ignoring them.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_BACKENDOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_BACKENDOPTIONS_TD

include "Common/BitcodeOptions.td"
include "Common/CompilationOptions.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def mlir_timing: Flag<["--", "-"], "mlir-timing">,
  Group<CompilationOptionGroup>, Index<18>, Flags<[HelpHidden]>,
  HelpText<
    "Times each MLIR pass and analysis, printing a report to stderr when "
    "compilation finishes, which for `mojo run` is before the program starts. "
    "Passes that the compilation cache serves from an earlier build never "
    "run, so a warm cache reports the parse and nothing else; point "
    "`MODULAR_CACHE_DIR` at an empty directory to time a whole pipeline.">,
  MarshallingInfoFlag<DriverOpts<"mlirTiming">>;

def llvm_timing: Flag<["--", "-"], "llvm-timing">,
  Group<CompilationOptionGroup>, Index<20>, Flags<[HelpHidden]>,
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

defm mlir_timing_display: CommandOptionHidden<["--", "-"],
  "mlir-timing-display",
  CompilationOptionGroup, Index<19>, MetaVarName<"MODE">,
  HelpText<
    "How `--mlir-timing` groups its report. `tree` (the default) nests "
    "timings by pipeline structure; `list` aggregates per pass name, sorted "
    "by total time. The `list` mode merges the passes of all pipelines, so it "
    "cannot show the time of one offload target on its own.">>,
  MarshallingInfoString<DriverOpts<"mlirTimingDisplay">>;

def timing_json: Flag<["--", "-"], "timing-json">,
  Group<CompilationOptionGroup>, Index<21>, Flags<[HelpHidden]>,
  HelpText<
    "Writes the timing that `--mlir-timing` and `--llvm-timing` collect as "
    "one JSON object rather than as text. The object holds a `mlir` member "
    "and an `llvm` member, and only the members the command asks for, so it "
    "is empty if the command asks for no timing. The `mlir` member keeps the "
    "tree or the list that `--mlir-timing-display` selects. The `llvm` member "
    "holds one entry for each pipeline, with the name of the pipeline and the "
    "times of its passes.">,
  MarshallingInfoFlag<DriverOpts<"timingJSON">>;

defm timing_file: CommandOptionHidden<["--", "-"], "timing-file",
  CompilationOptionGroup, Index<22>, MetaVarName<"FILE">,
  HelpText<
    "Writes the timing reports to FILE rather than to stderr. The reports are "
    "text, or JSON with `--timing-json`.">>,
  MarshallingInfoString<DriverOpts<"timingFile">>;

//===----------------------------------------------------------------------===//
// Linker Options
//===----------------------------------------------------------------------===//

def LinkOptionGroup : OptionGroup<"Linker options">, Index<15>;

defm Xlinker : CommandOption<["-"], "Xlinker",
  LinkOptionGroup, Index<1>, MetaVarName<"ARG">,
  HelpText<"Pass ARG to the linker.">>;

defm lld_path : CommandOption<["--", "-"], "lld-path",
  LinkOptionGroup, Index<2>, MetaVarName<"PATH">,
  HelpText<"Overrides the path to the `lld` linker used when linking. Takes "
           "precedence over the `MODULAR_MOJO_MAX_LLD_PATH` environment "
           "variable and the `mojo-max.lld_path` configuration value.">>,
  MarshallingInfoString<DriverOpts<"lldPath">>;

#endif // KGEN_TOOLS_MOJO_COMMON_BACKENDOPTIONS_TD
