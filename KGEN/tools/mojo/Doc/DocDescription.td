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
  "The input may be a single file or a directory. If you specify a "
  "directory, it will generate a single JSON output with documentation "
  "for all modules found in that path, recursively.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<7>;
