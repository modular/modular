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

def DiagnosticOptionGroup : OptionGroup<"Diagnostic options">, Index<10>;

def diagnose_missing_doc_strings : Flag<["--", "-"],
  "diagnose-missing-doc-strings">, Group<DiagnosticOptionGroup>, Index<0>,
  HelpText<"Emits diagnostics for missing or partial doc strings.">;

def validate_doc_strings : Flag<["--", "-"],
  "validate-doc-strings">, Group<DiagnosticOptionGroup>, Index<1>,
  HelpText<"Emits errors for invalid doc strings instead of warnings.">;

defm max_notes : CommandOption<["--", "-"], "max-notes-per-diagnostic",
  DiagnosticOptionGroup, Index<2>, MetaVarName<"INTEGER">,
  HelpText<"When the Mojo compiler emits diagnostics, it sometimes also prints "
           "notes with additional information. This option sets an upper "
           "threshold on the number of notes that can be printed with a "
           "diagnostic. If not specified, the default maximum is 10.">>;

#endif // KGEN_TOOLS_MOJO_COMMON_DIAGNOSTICOPTIONS_TD
