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

def FormatOptionGroup : OptionGroup<"Format options">, Index<0>;

defm line_length : CommandOption<["--", "-"], "line-length",
  FormatOptionGroup, Index<0>, MetaVarName<"INTEGER">,
  HelpText<"Sets the max character line length. Default is 80.">>;
def : Separate<["-"], "l">, Group<FormatOptionGroup>, Index<1>,
  MetaVarName<"INTEGER">, Alias<line_length>;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic options">, Index<1>;

def quiet : Flag<["--"], "quiet">, Group<DiagnosticOptionGroup>,
  Index<0>, HelpText<"Disables non-error messages.">;
def : Flag<["-"], "q">, Group<DiagnosticOptionGroup>, Index<0>, Alias<quiet>;

def print_cache_dir : Flag<["--"], "print-cache-dir">,
  Group<DiagnosticOptionGroup>, Index<1>,
  Flags<[HelpHidden]>,
  HelpText<"Print the cache directory path and exit.">;
