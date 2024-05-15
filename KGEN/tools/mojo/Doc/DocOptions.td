//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Doc/DocDescription.td"

include "Common/DiagnosticFormatOptions.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"Sets the path and filename for the JSON output. If not provided, "
           "output is written to stdout.">>;

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<1>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories that Mojo will "
           "search for any package/module dependencies. That is, if the file "
           "you pass to `mojo doc` imports any packages that do not reside in "
           "the local path and are not part of the Mojo standard library, use "
           "this to specify the path where Mojo can find those packages.">;

//===----------------------------------------------------------------------===//
// Validation Options
//===----------------------------------------------------------------------===//

def ValidationOptionGroup :
  OptionGroup<"Validation options">, Index<2>,
  HelpText<
    "The following validation options help ensure that your docstrings use "
    "valid structure and meet other style criteria. By default, warnings are "
    "emitted only if the docstrings contain errors that prevent translation to "
    "the output format. (More options coming later.)">;

def diagnose_missing_doc_strings : Flag<["--", "-"],
  "diagnose-missing-doc-strings">, Group<ValidationOptionGroup>, Index<0>,
  HelpText<"Emits diagnostics for missing or partial doc strings.">;

def validate_doc_strings : Flag<["--", "-"],
  "validate-doc-strings">, Group<ValidationOptionGroup>, Index<1>,
  HelpText<"Emits errors for invalid doc strings instead of warnings.">;
