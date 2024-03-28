//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def DocDescription : MojoCommandDescription<
  "doc", "Compiles docstrings from a Mojo file.",
  "This is an early version of a documentation tool that generates an API "
  "reference from Mojo code comments. Currently, it generates a structured "
  "output of all docstrings into a JSON file, and it does not generate HTML. "
  "This output format is subject to change.\n"
  "\n"
  "The input must be the path to a single Mojo source file.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<7>;
