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
  "package directory, in which you import each local Mojo module and its "
  "members that you want in the package (a module is any Mojo source file "
  "without a `main()` function). Then pass that package directory name to this "
  "command, and specify the output path and filename with `-o`. "
  "Once you have a `.mojopkg` or `.📦` file, you can import it into your "
  "project using that package name.\n"
  "\n"
  "For example, if your `mypackage/__init__.mojo` file looks like this:\n"
  "\n"
  "    from .mymodule import MyString\n"
  "\n"
  "That means `mypackage/mymodule.mojo` includes a struct named `MyString`. "
  "You can then package it with this command:\n"
  "\n"
  "    mojo package mypackage -o mypackage.mojopkg\n"
  "\n"
  "And then anywhere you place this package file, you can import `MyString` "
  "like this:\n"
  "\n"
  "    from mypackage import MyString\n"
  "\n"
  "And you're all set." ,
  [UsageInfo<"options", "path", /*variadicInput=*/false>]>, Index<3>;
