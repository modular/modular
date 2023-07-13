//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Package/PackageDescription.td"
include "Support/Driver/CommonOptions.td"

//===----------------------------------------------------------------------===//
// Compilation Options
//===----------------------------------------------------------------------===//

def CompilationOptionGroup : OptionGroup<"Compilation Options">, Index<0>;

def o : JoinedOrSeparate<["-"], "o">, Group<CompilationOptionGroup>, Index<0>,
  MetaVarName<"PATH">, HelpText<"The path to which an output file "
  "will be written. If not provided, output is written to stdout">;

def name : JoinedOrSeparate<["--"], "name">, Group<CompilationOptionGroup>,
  Index<1>, MetaVarName<"NAME">, HelpText<"The name to give to the new package."
  " This should be a FQDN that you own, if you plan to publish this package.">;

def no_optimization : Flag<["--", "-"], "no-optimization">,
  Group<CompilationOptionGroup>, Index<2>,
  HelpText<"Whether to disable compiler optimizations. This may reduce the "
           "amount of time it takes to compile the Mojo source file. It may "
           "also reduce the runtime performance of the compiled executable.">;
def : Flag<["-"], "O0">, Group<CompilationOptionGroup>, Index<0>,
  Alias<no_optimization>;

def debug_level : Separate<["--", "-"], "debug-level">,
  Group<CompilationOptionGroup>, Index<3>, MetaVarName<"LEVEL">,
  HelpText<"The level of debug info to use during compilation. The value must "
           "be one of: `none` (the default value), `line-tables`, `full`">;

def triple : JoinedOrSeparate<["--"], "triple">, Group<CompilationOptionGroup>,
  Index<4>, MetaVarName<"TRIPLE">, HelpText<"The target triple to use to "
  "compile this package">;

def cpu : JoinedOrSeparate<["--"], "cpu">, Group<CompilationOptionGroup>,
  Index<5>, MetaVarName<"CPU">, HelpText<"The target CPU to use to compile "
  "this package.">;

def features : JoinedOrSeparate<["--"], "features">,
  Group<CompilationOptionGroup>, Index<6>, MetaVarName<"FEATURES">,
  HelpText<"The target features to use to compile this package.">;
