//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Doc/DocDescription.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

def o : JoinedOrSeparate<["-"], "o">, Group<OutputOptionGroup>, Index<0>,
  MetaVarName<"PATH">, HelpText<"The path to which an output file "
  "will be written. If not provided, output is written to stdout.">;

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<1>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"Append the given path to the list of directories to search for "
           "included Mojo files.">;

//===----------------------------------------------------------------------===//
// Validation Options
//===----------------------------------------------------------------------===//

def ValidationOptionGroup :
  OptionGroup<"Validation options">, Index<2>,
  HelpText<
    "During translation of doc strings, this tool may emit warnings if those "
    "doc strings do not meet certain criteria. By default, warnings are "
    "emitted if the doc strings contain errors that prevent translation to the "
    "output format, but these options can be used to enable or disable these "
    "and other validation warnings.">;

def warn_missing_dog_strings : Flag<["--", "-"], "warn-missing-doc-strings">,
  Group<ValidationOptionGroup>, Index<0>,
  HelpText<"Emit warnings for missing or partial doc strings.">;
