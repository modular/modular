//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a common set of diagnostic options for setting up the
// compilation environment for Mojo programs.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICOPTIONS_TD
#define KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICOPTIONS_TD

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
  HelpText<"Emits diagnostics for missing or partial doc strings.">;

def validate_doc_strings : Flag<["--", "-"],
  "validate-doc-strings">, Group<DiagnosticOptionGroup>, Index<2>,
  HelpText<"Emits errors for invalid doc strings instead of warnings.">;

defm max_notes : CommandOption<["--", "-"], "max-notes-per-diagnostic",
  DiagnosticOptionGroup, Index<3>, MetaVarName<"INTEGER">,
  HelpText<"When the Mojo compiler emits diagnostics, it sometimes also prints "
           "notes with additional information. This option sets an upper "
           "threshold on the number of notes that can be printed with a "
           "diagnostic. If not specified, the default maximum is 10.">>;

defm strip_file_prefix : CommandOptionHidden<["-"], "strip-file-prefix",
  DiagnosticOptionGroup, Index<4>, MetaVarName<"PATH">,
  HelpText<
    "Strip this prefix from filenames used for diagnostics & debugging.">>;

def disable_builtins : Flag<["--", "-"],
  "disable-builtins">, Group<DiagnosticOptionGroup>, Index<5>,
  HelpText<"Do not use builtins when create package.">;

def mojo_search_paths : JoinedOrSeparate<["-"], "mojo-search-paths">,
    Group<DiagnosticOptionGroup>, Index<6>,
    MetaVarName<"PATH">,
    HelpText<"Specify additional search paths for Mojo modules.">,
    Flags<[HelpHidden]>;

def disable_warnings : Flag<["--", "-"],
  "disable-warnings">, Group<DiagnosticOptionGroup>, Index<7>,
  HelpText<"Do not print warning messages.">;

#endif // KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICOPTIONS_TD
