//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def DocDescription : MojoCommandDescription<
  "doc", "Compiles doc strings",
  "Translate Mojo source file doc strings to a structured output format.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<5>;
