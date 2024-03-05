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
  "For more information, see [Mojo modules and "
  "packages](https://docs.modular.com/mojo/manual/packages)." ,
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<4>;
