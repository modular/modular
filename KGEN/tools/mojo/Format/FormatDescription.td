//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def FormatDescription : MojoCommandDescription<
  "format", "Format Mojo source files",
  "Format the given set of Mojo sources.",
  [UsageInfo<"options", "sources", /*variadicInput=*/true>]>, Index<4>;
