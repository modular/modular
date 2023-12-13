//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines options for the `greeter-cli hi` subcommand.
// See `../DriverOptions.td` for more explanations of the concepts here.
//
//===----------------------------------------------------------------------===//

include "Hi/HiDescription.td"
include "Support/Driver/CommonOptions.td"

def HiOptionGroup : OptionGroup<"Hi options">, Index<0>;

def hello : Flag<["--", "-"], "hello">, Group<HiOptionGroup>, Index<0>,
  HelpText<"Uses a more formal \"hello\" when printing a greeting.">;
