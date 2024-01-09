//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "DescriptionBase.td"

def PackageDescription : MojoCommandDescription<
  "package", "Compiles a Mojo package.",
  "Compiles a directory of Mojo source files into a binary package suitable "
  "to share and import into other Mojo programs and modules.\n"
  "\n"
  "To create a Mojo package, first add an `__init__.mojo` file to your "
  "package directory. Then pass that directory name to this "
  "command, and specify the output path and filename with `-o`.\n"
  "\n"
  "By default, packages are compiled only as much as is possible to do in a "
  "machine-agnostic way. Then, they're fully compiled when a user of the "
  "package first imports it within a Mojo program on their own machine. "
  "However, by specifying certain target options, such as `--target-triple`, "
  "packages will be fully compiled in advance.\n"
  "\n"
  "For more information, see [Mojo modules and "
  "packages](https://docs.modular.com/mojo/manual/get-started/packages.html)." ,
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<4>;
