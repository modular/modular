//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "Package/PackageDescription.td"
include "Support/Driver/CommonOptions.td"
include "../Common/CompilationOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

def o : JoinedOrSeparate<["-"], "o">, Group<OutputOptionGroup>, Index<0>,
  MetaVarName<"PATH">, HelpText<"The path to which an output file "
  "will be written. If not provided, output is written to stdout.">;

def name : JoinedOrSeparate<["--"], "name">, Group<OutputOptionGroup>,
  Index<1>, MetaVarName<"NAME">, HelpText<"The name to give to the new package."
  " This should be a FQDN that you own, if you plan to publish this package.">;
