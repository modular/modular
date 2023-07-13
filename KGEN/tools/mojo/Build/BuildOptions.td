//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "BuildDescription.td"
include "Support/Driver/CommonOptions.td"
include "../Common/CompilationOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output Options">, Index<0>;

def o : JoinedOrSeparate<["-"], "o">, Group<OutputOptionGroup>, Index<0>,
  MetaVarName<"PATH">, HelpText<"The path to which an output file "
  "will be written">;
