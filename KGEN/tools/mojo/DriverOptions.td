//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DriverDescription.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def DiagnosticOptionGroup : OptionGroup<"Diagnostic options">, Index<0>;

def version : Flag<["--", "-"], "version">, Group<DiagnosticOptionGroup>,
  Index<0>, HelpText<"Prints the Mojo version and exits.">;

def v : Flag<["-"], "v">, Group<DiagnosticOptionGroup>, Alias<version>,
  Index<0>;

def print_cache_location : Flag<["--", "-"], "print-cache-location">,
  Group<DiagnosticOptionGroup>, Index<0>,
  HelpText<"Prints the Mojo compile cache (.mojo_cache) location and exits.">;
