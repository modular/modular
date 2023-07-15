//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Format/FormatDescription.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Format Options
//===----------------------------------------------------------------------===//

def FormatOptionGroup : OptionGroup<"Format Options">, Index<0>;

def line_length : Separate<["--"], "line-length">,
  Group<FormatOptionGroup>, Index<0>, MetaVarName<"INTEGER">,
  HelpText<"How many characters per line to allow, with a default of 80.">;
def : Separate<["-"], "l">, Group<FormatOptionGroup>, Index<0>,
  MetaVarName<"INTEGER">, Alias<line_length>;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic Options">, Index<1>;

def quiet : Flag<["--"], "quiet">, Group<DiagnosticOptionGroup>,
  Index<0>, HelpText<"Don't emit non-error messages to stderr.">;
def : Flag<["-"], "q">, Group<DiagnosticOptionGroup>, Index<0>, Alias<quiet>;
