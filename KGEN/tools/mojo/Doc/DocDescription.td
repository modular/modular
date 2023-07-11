//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def DocDescription : MojoCommandDescription<
  "doc", "Compile doc strings",
  "Translate Mojo source file doc strings to a structured output format.",
  /*inputMetaVarName=*/"path", /*requiresInput=*/true, /*variadicInput=*/false>,
  Index<3>;
