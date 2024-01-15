//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "BuildDescription.td"
include "Support/Driver/CommonOptions.td"
include "../Common/CompilationOptions.td"
include "../Common/TargetOptions.td"

//===----------------------------------------------------------------------===//
// Output Options
//===----------------------------------------------------------------------===//

def OutputOptionGroup : OptionGroup<"Output options">, Index<0>;

defm o : CommandOption<["-"], "o", OutputOptionGroup, Index<0>,
  MetaVarName<"PATH">, HelpText<
    "Sets the path and filename for the executable output. By default, it "
    "outputs the executable to the same location as the Mojo file, with the "
    "same name and no extension."
  >>;
