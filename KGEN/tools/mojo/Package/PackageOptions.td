//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Package/PackageDescription.td"

include "Common/DiagnosticFormatOptions.td"
include "Common/DiagnosticOptions.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation options">, Index<5>;

def I : JoinedOrSeparate<["-"], "I">, Group<CompilationOptionGroup>, Index<1>,
  MetaVarName<"PATH">,
  HelpText<"Appends the given path to the list of directories to search for "
           "imported Mojo files.">;

def kgenModule : Flag<["-"], "kgenModule">, Group<CompilationOptionGroup>,
  Index<2>,
  HelpText<"Export as a KGEN module.">;

defm bitcode_libs : CommandOptionHidden<["--", "-"], "bitcode-libs",
  CompilationOptionGroup, Index<3>, MetaVarName<"LIBS">,
  HelpText<
    "External bitcode libraries to package into the generated package."
  >>;

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">, HelpText<
    "Sets the path and filename for the output package. The filename must end "
    "with either `.mojopkg` or `.📦`. The filename given here defines the "
    "package name you can then use to import the code (minus the file "
    "extension). If you don't specify this option, a `.mojopkg` file is "
    "generated in the current working directory, with a name based on the name "
    "of the input directory."
  >>;
