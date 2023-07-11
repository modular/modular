//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def DemangleDescription : MojoCommandDescription<
  "demangle", "Demangle the given name",
  "If the given name is a mangled Mojo symbol name, prints the demangled "
  "name. If no name is provided, one is read from standard input.",
  /*inputMetaVarName=*/"name", /*requiresInput=*/false,
  /*variadicInput=*/false>, Index<4>;
