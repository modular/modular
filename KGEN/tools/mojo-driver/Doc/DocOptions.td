//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "../HelpOptions.td"

def MojoDocGroup : OptionGroup<
  "Options specific to the doc subcommand.">;

def o : JoinedOrSeparate<["-"], "o">, Group<MojoDocGroup>, Flags<[]>,
  MetaVarName<"<path>">,
  HelpText<"The path to which an output file will be written. If not "
           "provided, output is written to stdout.">;

def I : JoinedOrSeparate<["-"], "I">, Group<MojoDocGroup>, Flags<[]>,
  MetaVarName<"<path>">,
  HelpText<"Append the given path to the list of directories to search for "
           "included Mojo files.">;

def validate : Flag<["--", "-"], "validate">, Group<MojoDocGroup>, Flags<[]>,
  HelpText<"Validate doc strings as they are parsed. When enabled, warning "
           "diagnostics are emitted as invalid doc strings are parsed.">;
