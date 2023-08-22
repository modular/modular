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
  MetaVarName<"PATH">, HelpText<"Sets the path and filename for the output "
  "package. If not provided, output is written to stdout.">;

def name : JoinedOrSeparate<["--"], "name">, Group<OutputOptionGroup>,
  Index<1>, MetaVarName<"NAME">, HelpText<"Sets the package name. This is the "
  "name that you and others will use to import the package, and it may be "
  "different from the actual package filename. If you plan to publish this "
  "package, this name should be a fully-qualified domain name that you own.">;
