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
// This file defines a common set of diagnostic options for setting up the
// compilation environment for Mojo programs.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICOPTIONS_TD

include "Common/Marshalling.td"
include "Support/Driver/DriverCommand.td"

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Compilation diagnostic options">,
  Index<10>,
  HelpText<"Controls how the Mojo compiler outputs diagnostics related to "
           "compiling and running Mojo source code.">;

def diagnose_missing_doc_strings : Flag<["--", "-"],
  "diagnose-missing-doc-strings">, Group<DiagnosticOptionGroup>, Index<1>,
  HelpText<"Emits diagnostics for missing or partial doc strings.">,
  MarshallingInfoFlag<FrontendOpts<"diagnoseMissingDocStrings">>;

defm max_notes : CommandOption<["--", "-"], "max-notes-per-diagnostic",
  DiagnosticOptionGroup, Index<3>, MetaVarName<"INTEGER">,
  HelpText<"When the Mojo compiler emits diagnostics, it sometimes also prints "
           "notes with additional information. This option sets an upper "
           "threshold on the number of notes that can be printed with a "
           "diagnostic. If not specified, the default maximum is 10.">>,
  MarshallingInfoInt<FrontendOpts<"maxNotesPerDiagnostic">, "10", "int">;

defm strip_file_prefix : CommandOptionHidden<["-"], "strip-file-prefix",
  DiagnosticOptionGroup, Index<4>, MetaVarName<"PATH">,
  HelpText<
    "Strip this prefix from filenames used for diagnostics & debugging.">>,
  MarshallingInfoString<FrontendOpts<"stripFilePrefix">>;

def disable_builtins : Flag<["--", "-"],
  "disable-builtins">, Group<DiagnosticOptionGroup>, Index<5>,
  HelpText<"Do not use builtins when create package.">,
  MarshallingInfoNegativeFlag<FrontendOpts<"useBuiltinModule">>;

def mojo_search_paths : JoinedOrSeparate<["-"], "mojo-search-paths">,
    Group<DiagnosticOptionGroup>, Index<6>,
    MetaVarName<"PATH">,
    HelpText<"Specify additional search paths for Mojo modules, as a "
             "comma-separated list. May be given more than once.">,
    Flags<[HelpHidden]>,
    MarshallingInfoCustom<FrontendOpts<"searchPaths">,
                          "normalizeCommaJoinedList">;
def : Joined<["-"], "mojo-search-paths=">, Group<DiagnosticOptionGroup>,
  Index<0>, Alias<mojo_search_paths>, Flags<[HelpHidden]>;

def disable_warnings : Flag<["--", "-"],
  "disable-warnings">, Group<DiagnosticOptionGroup>, Index<7>,
  HelpText<"Do not print warning messages.">;

def werror : Flag<["--", "-"], "Werror">,
  Group<DiagnosticOptionGroup>, Index<10>,
  HelpText<"Treat warnings as errors.">;

def wno_error : Flag<["--", "-"], "Wno-error">,
  Group<DiagnosticOptionGroup>, Index<11>,
  HelpText<"Do not treat warnings as errors.">;

def warn_on_unstable_apis : Flag<["--", "-"], "warn-on-unstable-apis">,
  Group<DiagnosticOptionGroup>, Index<12>,
  HelpText<"Warn when using unstable APIs from the standard library.">,
  MarshallingInfoFlag<FrontendOpts<"warnOnUnstableAPIs">>;

def ignore_incompatible_precompiled_file_errors : Flag<["--"], "ignore-incompatible-precompiled-file-errors">,
  Group<DiagnosticOptionGroup>, Index<13>,
  HelpText<"Ignore errors encountered when loading incompatible Mojo precompiled files.">,
  MarshallingInfoFlag<FrontendOpts<"ignoreIncompatiblePrecompiledFileErrors">>;

defm ignore_deprecated : CommandOption<["--", "-"], "ignore-deprecated",
  DiagnosticOptionGroup, Index<14>, MetaVarName<"NAME">,
  HelpText<"Suppress the deprecation warning for the given declaration (e.g. "
           "`Foo.bar`, or `some_fn` for a top-level declaration).">>,
  MarshallingInfoStringVector<FrontendOpts<"ignoredDeprecations">>;

#endif // KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICOPTIONS_TD
