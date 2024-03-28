//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def FormatDescription : MojoCommandDescription<
  "format", "Formats Mojo source files.",
  "Formats the given set of Mojo sources using a Mojo-specific lint tool.",
  [UsageInfo<"options", "sources", /*variadicInput=*/true>]>, Index<6>;
