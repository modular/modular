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
  "package. The filename should end with either `.mojopkg` or `.📦`. If not "
  "provided, output is written to stdout.">;
