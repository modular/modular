//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "../OptionsBase.td"
include "../CommonOptions.td"

//===----------------------------------------------------------------------===//
// doc Command
//===----------------------------------------------------------------------===//

def DocDescription : MojoCommandDescription<
  "doc", "Compile doc strings",
  "Translate Mojo source file doc strings to a structured output format.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/false>;

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation Options">, Index<0>;

def o : JoinedOrSeparate<["-"], "o">, Group<CompilationOptionGroup>, Index<0>,
  MetaVarName<"PATH">,
  HelpText<"The path to which an output file will be written. If not "
           "provided, output is written to stdout.">;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<1>,
  MetaVarName<"PATH">,
  HelpText<"Append the given path to the list of directories to search for "
           "included Mojo files.">;

//===----------------------------------------------------------------------===//
// Validation Options
//===----------------------------------------------------------------------===//

def ValidationOptionGroup :
  OptionGroup<"Validation Options">, Index<1>,
  HelpText<
    "During translation of doc strings, this tool may emit warnings if those "
    "doc strings do not meet certain criteria. By default, warnings are "
    "emitted if the doc strings contain errors that prevent translation to the "
    "output format, but these options can be used to enable or disable these "
    "and other validation warnings.">;

def validate : Flag<["--", "-"], "validate">,
  Group<ValidationOptionGroup>, Index<0>,
  HelpText<"Validate doc strings as they are parsed. When enabled, warning "
           "diagnostics are emitted as invalid doc strings are parsed.">;
