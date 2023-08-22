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
  "To create a Mojo package, first add an `__init__.mojo` file to the "
  "package directory, in which you import each local Mojo module that you "
  "want in the package (a module is any Mojo source file without a `main()` "
  "function). Then pass that directory name to this command, and specify the "
  "package name with `--name` and the output path with `-o`.",
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<3>;
