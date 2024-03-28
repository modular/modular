//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def DemangleDescription : MojoCommandDescription<
  "demangle", "Demangles the given name.",
  "If the given name is a mangled Mojo symbol name, prints the demangled "
  "name. If no name is provided, one is read from standard input.",
  [UsageInfo<"options", "name", /*variadicInput=*/false>]>, Index<8>;
