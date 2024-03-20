//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def TestDescription : MojoCommandDescription<
  "test", "Execute unit, integration, and documentation tests.",
  "Execute the given Mojo tests.",
  [UsageInfo<"options", "input", /*variadicInput=*/false>]>, Index<8>;
