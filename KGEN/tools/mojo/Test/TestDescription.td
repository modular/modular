//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def TestDescription : MojoCommandDescription<
  "test", "DEPRECATED. Execute unit, integration, and documentation tests.",
  "WARNING: This command is deprecated and will be removed in the future. See https://github.com/modular/modular/blob/main/mojo/proposals/mojo-test-deprecation.md for more information.",
  [UsageInfo<"options", "input", /*variadicInput=*/false>]>, Index<9>;
