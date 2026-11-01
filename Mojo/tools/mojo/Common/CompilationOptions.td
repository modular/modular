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
// This file defines a common set of compilation options for setting up the
// compilation environment for Mojo programs. The index of the option groups is
// set to multiples of 5, to allow other option groups to be inserted before or
// after each.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD

include "Common/IncludeOptions.td"
include "Common/Marshalling.td"
include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

defm optimization_level : CommandOption<["--", "-"], "optimization-level",
  CompilationOptionGroup, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of optimization to use at compilation. The value must be "
    "a number between 0 and 3. The default is 3."
  >>,
  MarshallingInfoCustom<CompilationOpts<"optimizationLevel">,
                        "normalizeOptimizationLevel">;
def : JoinedOrSeparate<["-"], "O">, Group<CompilationOptionGroup>,
  Alias<optimization_level>;
def : Flag<["--", "-"], "no-optimization">, Group<CompilationOptionGroup>,
  Alias<optimization_level>, AliasArgs<["0"]>;

def D : JoinedOrSeparate<["-"], "D">, Group<CompilationOptionGroup>,
  MetaVarName<"KEY=VALUE">,
  HelpText<"Defines a named value that can be used from within the Mojo source "
           "file being executed. For example, `-Dfoo=42` defines a name `foo` "
           "that, when queried with the `std.defines` module from within the "
           "Mojo program, would yield the compile-time value `42`.">,
  MarshallingInfoStringVector<DriverOpts<"defines">>;

defm debug_level : CommandOption<["--", "-"], "debug-level",
  CompilationOptionGroup, MetaVarName<"LEVEL">,
  HelpText<
    "Sets the level of debug info to use at compilation. The value must be one "
    "of: `none`, `line-tables`, or `full`. Default is `none`, except when "
    "using `mojo debug foo.mojo`, which defaults to `full`. Please note that "
    "there are issues when generating debug info for some Mojo programs that "
    "have yet to be addressed."
  >>,
  MarshallingInfoChoice<CompilationOpts<"debugLevel">,
                        ["none", "line-tables", "full"],
                        ["kNoDebug", "kLineTablesOnly", "kFullDebugInfo"]>;

def : Flag<["-"], "g">, Group<CompilationOptionGroup>,
  Alias<debug_level>, AliasArgs<["full"]>;

def : Flag<["-"], "g0">, Group<CompilationOptionGroup>,
  Alias<debug_level>, AliasArgs<["none"]>;

def : Flag<["-"], "g1">, Group<CompilationOptionGroup>,
  Alias<debug_level>, AliasArgs<["line-tables"]>;

def : Flag<["-"], "g2">, Group<CompilationOptionGroup>,
  Alias<debug_level>, AliasArgs<["full"]>;

defm num_threads : CommandOption<["--", "-"], "num-threads",
  CompilationOptionGroup, MetaVarName<"NUM">,
  HelpText<
    "Sets the maximum number of threads to use for compilation. "
    "The default is 0 (use all available threads)."
  >>,
  MarshallingInfoInt<CompilationOpts<"numThreads">, "0", "size_t">;

def : JoinedOrSeparate<["-"], "j">, Group<CompilationOptionGroup>,
  Alias<num_threads>;

defm loop_unrolling_warn_threshold: CommandOptionHidden<["--", "-"],
  "loop-unrolling-warn-threshold",
  CompilationOptionGroup, MetaVarName<"LUWT">,
  HelpText<"Sets loop unrolling factor warning threshold, "
           "default is 1024, set to 0 to disable the warning.">>,
  MarshallingInfoInt<CompilationOpts<"loopUnrollingWarnThreshold">, "1024",
                     "int64_t">;

defm elaboration_error_limit: CommandOptionHidden<["--", "-"],
  "elaboration-error-limit",
  CompilationOptionGroup, MetaVarName<"ELABERRORLIMIT">,
  HelpText<
    "Sets limit to number of produced errors during elaboration "
    "before stop emitting diagnostics. The default is 20, and "
    "the limit can be disabled with -elaboration-error-limit=0.">>,
  MarshallingInfoInt<CompilationOpts<"elaborationErrorLimit">, "20", "int">;

def elaboration_error_include_prelude: Flag<["--", "-"],
  "elaboration-error-include-prelude">,
  Group<CompilationOptionGroup>,
  HelpText<
    "Show elaboration error with locations in mojo startup modules "
    "(prelude).">,
  MarshallingInfoFlag<CompilationOpts<"elaborationErrorIncludePrelude">>;

defm elaboration_error_verbose: CommandOptionHidden<["--", "-"],
  "elaboration-error-verbose",
  CompilationOptionGroup, MetaVarName<"ELABERRORVERBOSE">,
  HelpText<
    "Controls how parameter values are displayed as part of "
    "elaboration errors when function instantiation fails. "
    "The default is `simple-params`, which "
    "display concretized parameter values for "
    "simple types, including numeric types and "
    "strings, in a user-friendly format; "
    "`no-params` means don't display any concrete parameter values; "
    "`all-params` means show all concrete parameter values.">>,
  MarshallingInfoChoice<CompilationOpts<"elaborationErrorVerbose">,
                        ["no-params", "simple-params", "all-params"],
                        ["kNoParams", "kSimpleParams", "kAllParams"]>;

defm elaboration_max_depth: CommandOptionHidden<["--", "-"],
  "elaboration-max-depth",
  CompilationOptionGroup, MetaVarName<"ELABMAXDEPTH">,
  HelpText<
      "The maximum elaborator instantiation depth. Default is "
      "std::numeric_limits<unsigned>::max(). Elaborator errors out if "
      "the depth of compile time recursion exceeds this value.">>,
  MarshallingInfoInt<CompilationOpts<"elaborationMaxDepth">>;

defm fp_mode : CommandOption<["--", "-"], "fp-mode",
  CompilationOptionGroup, MetaVarName<"MODE">,
  HelpText<
    "Controls floating-point behavior as a comma-separated list of "
    "`feature=value` items (may be given more than once). The only feature is "
    "`contract`, one of `fast` (default) or `off`. `contract=fast` is like "
    "Clang's `-ffp-contract=fast`: it fuses `a + b*c` into an FMA across "
    "statements and breaking strict IEEE compliance. "
    "`contract=off` disables contraction."
  >>,
  MarshallingInfoCustom<CompilationOpts<"fpMode">, "normalizeFpMode">;

//===----------------------------------------------------------------------===//
// Experimental Compilation Options
//===----------------------------------------------------------------------===//

def ExperimentalCompilationOptionGroup :
  OptionGroup<"Experimental compilation options">, Index<16>;

defm sanitize : CommandOption<["--", "-"], "sanitize",
  ExperimentalCompilationOptionGroup, MetaVarName<"CHECK">,
  HelpText<
    "Turns on runtime checks. The following values are supported: `address` "
    "(detects memory issues), and `thread` (detects multi-threading issues)."
  >>,
  MarshallingInfoCustom<CompilationOpts<"sanitizers">, "normalizeSanitizer">;

def shared_libasan : Flag<["--", "-"], "shared-libasan">,
  Group<ExperimentalCompilationOptionGroup>,
  HelpText<
    "Dynamically link the address sanitizer runtime. Requires address "
    "sanitization turned on with `--sanitize` option."
  >,
  MarshallingInfoFlag<CompilationOpts<"sharedLibasan">>;

defm external_libasan : CommandOptionHidden<["--", "-"], "external-libasan",
  ExperimentalCompilationOptionGroup, MetaVarName<"FILE">,
  HelpText<
    "Uses the provided file as libasan instead of the default provided by "
    "the system. This file must provide a compatible interface as libasan, "
    "and must be present at both compile time and run time of the program."
  >>,
  MarshallingInfoString<CompilationOpts<"externalLibasan">>;

defm debug_info_language : CommandOption<["--", "-"], "debug-info-language",
  ExperimentalCompilationOptionGroup, MetaVarName<"LANGUAGE">,
  HelpText<
    "Sets the language to emit as part of the debug info. The supported "
    "languages are: `Mojo`, and `C`. `Mojo` is the default. `C` is useful to "
    "enable rudimentary debugging and binary introspection in tools "
    "that don't understand Mojo, but is not required for `mojo debug`."
  >>,
  MarshallingInfoChoice<CompilationOpts<"debugInfoLanguage">, ["Mojo", "C"],
                        ["kLangMojo", "kLangC"]>;

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATIONOPTIONS_TD
