//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "OptionsBase.td"
include "CommonOptions.td"

//===----------------------------------------------------------------------===//
// Top-Level Command
//===----------------------------------------------------------------------===//

def : MojoDriverDescription<
  "The Mojo🔥 command line interface", "Interface with Mojo's various features.",
  /*metaVarName=*/"command", /*requiresInput=*/false, /*variadicInput=*/false>;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic Options">, Index<0>;

def version : Flag<["--", "-"], "version">, Group<DiagnosticOptionGroup>,
  Index<0>, HelpText<"Prints the Mojo version and exits.">;

def v : Flag<["-"], "v">, Group<DiagnosticOptionGroup>, Alias<version>,
  Index<0>;
