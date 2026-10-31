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

include "Common/CompilationOptions.td"
include "Common/DeveloperOptions.td"
include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Common/IncrParsingOptions.td"
include "Common/BitcodeOptions.td"
include "Common/TargetOptions.td"
include "Support/Driver/CommonOptions.td"

def KGENDescription : DriverDescription<
  "kgen", "Compiles and runs KGEN and Mojo programs.",
  "A testing tool that runs one command on its input: elaborating it, "
  "emitting it in one of the formats listed below, executing its funcs, "
  "checking it as the language server does, or precompiling a package.",
  [UsageInfo<"options", "file", /*variadicInput=*/true>]>;

// A field of the `AsyncRT::CPUDeviceOptions cpuDeviceOptions` being built.
class RuntimeOpts<string member>
    : KeyPathAndMacro<"cpuDeviceOptions.", member, "MOJO_">;

// A field of kgen's `CLOptions clOptions` being built.
class KGENOpts<string member>
    : KeyPathAndMacro<"clOptions.", member, "MOJO_">;

//===----------------------------------------------------------------------===//
// Commands
//===----------------------------------------------------------------------===//

def CommandOptionGroup : OptionGroup<"Commands">, Index<0>,
  HelpText<"Exactly one command must be given.">;

def elaborate : Flag<["--", "-"], "elaborate">,
  Group<CommandOptionGroup>,
  HelpText<"Elaborates the input.">;
def elaborate_eq : Joined<["--", "-"], "elaborate=">,
  Group<CommandOptionGroup>, MetaVarName<"MODE">,
  HelpText<"Elaborates the input with (`use-parametric-interpreter`) or "
           "without (`no-use-parametric-interpreter`) the parametric "
           "interpreter.">;

def emit : Flag<["--", "-"], "emit">,
  Group<CommandOptionGroup>,
  HelpText<"Emits funcs as an object file.">;
def emit_eq : Joined<["--", "-"], "emit=">,
  Group<CommandOptionGroup>, MetaVarName<"KIND">,
  HelpText<"Emits funcs as the given kind of output file. The kinds are "
           "listed at the end of this help.">;

def execute : Flag<["--", "-"], "execute">,
  Group<CommandOptionGroup>,
  HelpText<"Executes the funcs given by `--func`.">;

def lsp : Flag<["--", "-"], "lsp">,
  Group<CommandOptionGroup>,
  HelpText<"Processes the input as the language server does: a lazy parse and "
           "the check pipeline, printing the checked IR to stdout and the "
           "diagnostics to stderr.">;
def lsp_eq : Joined<["--", "-"], "lsp=">,
  Group<CommandOptionGroup>, MetaVarName<"MODE">,
  HelpText<"Processes the input as `--lsp` does. The only mode, `no-dump`, "
           "skips printing the checked IR.">;

def precompile : Flag<["--", "-"], "precompile">,
  Group<CommandOptionGroup>,
  HelpText<"Precompiles the Mojo package directory given as the input into a "
           "`.mojoc` file, or prints the package with `-S`.">;

defm func : CommandOption<["--", "-"], "func", CommandOptionGroup,
  MetaVarName<"NAME[:SIGNATURE]">,
  HelpText<"Names a func for `--execute` to run, with an optional signature "
           "such as `f32()`. May be given more than once.">>;

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<1>;

defm o : CommandOption<["-"], "o", OutputOptionGroup,
  MetaVarName<"PATH">,
  HelpText<"Sets the output path. Defaults to stdout.">>,
  MarshallingInfoString<KGENOpts<"outputFilename">>;

def S : Flag<["-"], "S">, Group<OutputOptionGroup>,
  HelpText<"Writes MLIR output in textual form instead of as bytecode.">,
  MarshallingInfoFlag<KGENOpts<"emitTextualAsm">>;

defm d : CommandOption<["-"], "d", OutputOptionGroup,
  MetaVarName<"PATH">,
  HelpText<"Writes a dependency file for the output to PATH.">>,
  MarshallingInfoString<KGENOpts<"dependencyFilename">>;

def save_temps : Flag<["--", "-"], "save-temps">,
  Group<OutputOptionGroup>,
  HelpText<"Keeps the intermediate files in the directory given by "
           "`--temps-dir`, which defaults to the output directory.">,
  MarshallingInfoFlag<KGENOpts<"saveTemps">>;

defm temps_dir : CommandOption<["--", "-"], "temps-dir", OutputOptionGroup,
  MetaVarName<"PATH">,
  HelpText<"Sets the directory for the intermediate files that `--save-temps` "
           "keeps.">>,
  MarshallingInfoString<KGENOpts<"tempsDir">>;

//===----------------------------------------------------------------------===//
// Testing Options
//===----------------------------------------------------------------------===//

def TestingOptionGroup : OptionGroup<"Testing options">, Index<20>;

def verify_diagnostics : Flag<["--", "-"], "verify-diagnostics">,
  Group<TestingOptionGroup>,
  HelpText<"Checks the emitted diagnostics against the `expected-*` comments "
           "in the input.">,
  MarshallingInfoFlag<KGENOpts<"verifyDiagnostics">>;

def ignore_failure : Flag<["--", "-"], "ignore-failure">,
  Group<TestingOptionGroup>,
  HelpText<"Prints the errors from executing funcs without failing.">,
  MarshallingInfoFlag<KGENOpts<"ignoreFailures">>;

def version : Flag<["--", "-"], "version">,
  Group<TestingOptionGroup>,
  HelpText<"Prints the version, the host target and the registered targets.">;

//===----------------------------------------------------------------------===//
// Runtime Options
//===----------------------------------------------------------------------===//

def RuntimeOptionGroup : OptionGroup<"Runtime options">, Index<22>;

defm workqueue : CommandOption<["--", "-"], "workqueue", RuntimeOptionGroup,
  MetaVarName<"TYPE">,
  HelpText<"Sets the work queue type: `single-thread` or `thread-pool` (the "
           "default).">>,
  MarshallingInfoChoice<RuntimeOpts<"workQueueType">,
                        ["single-thread", "thread-pool"],
                        ["kSingleThread", "kThreadPool"]>;

defm allocator : CommandOption<["--", "-"], "allocator", RuntimeOptionGroup,
  MetaVarName<"TYPE">,
  HelpText<"Sets the allocator type: `malloc`, `tcmalloc`, `leak-checker`, "
           "`profiler` or `use-after-free`.">>,
  MarshallingInfoChoice<RuntimeOpts<"allocatorType">,
                        ["malloc", "tcmalloc", "leak-checker", "profiler",
                         "use-after-free"],
                        ["kMalloc", "kTCMalloc", "kLeakChecker", "kProfiler",
                         "kUseAfterFree"]>;

defm max_threads : CommandOption<["--", "-"], "max-threads",
  RuntimeOptionGroup, MetaVarName<"NUM">,
  HelpText<"Caps the number of work queue threads chosen by heuristics.">>,
  MarshallingInfoInt<RuntimeOpts<"maxThreads">, "0", "size_t">;

defm thread_busy_wait_time_us : CommandOption<["--", "-"],
  "thread-busy-wait-time-us", RuntimeOptionGroup,
  MetaVarName<"MICROSECONDS">,
  HelpText<"Sets how long the work queue threads spin before sleeping. Zero "
           "means they never spin.">>,
  MarshallingInfoInt<RuntimeOpts<"threadBusyWaitTime">, "200", "size_t">;

defm time_profile : CommandOption<["--", "-"], "time-profile",
  RuntimeOptionGroup, MetaVarName<"BASE">,
  HelpText<"Writes the runtime's profile to `BASE.time-trace` and its events "
           "to `BASE.time-events.csv`, in builds with runtime profiling.">>,
  MarshallingInfoString<RuntimeOpts<"profileFilename">>;

defm profiler_debuginfo : CommandOption<["--", "-"], "profiler-debuginfo",
  RuntimeOptionGroup, MetaVarName<"KIND">,
  HelpText<"Emits debug info for a profiler: `none` (the default), `perf`, or "
           "`so`, which loads the compiled kernels from a shared library.">>,
  MarshallingInfoChoice<RuntimeOpts<"profilerDebuginfo">,
                        ["none", "perf", "so"],
                        ["kNoProfiler", "kPerfProfiler", "kSOProfiler"]>;
