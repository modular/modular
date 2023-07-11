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

def triple : JoinedOrSeparate<["--"], "triple">, Group<CompilationOptionGroup>,
  Index<2>, MetaVarName<"TRIPLE">, HelpText<"The target triple to use to "
  "compile this package">;

def cpu : JoinedOrSeparate<["--"], "cpu">, Group<CompilationOptionGroup>,
  Index<3>, MetaVarName<"CPU">, HelpText<"The target CPU to use to compile "
  "this package.">;

def features : JoinedOrSeparate<["--"], "features">,
  Group<CompilationOptionGroup>, Index<4>, MetaVarName<"FEATURES">,
  HelpText<"The target features to use to compile this package.">;
