//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes that can be used to document command line
// executables (or "subcommands" of a top-level driver executable).
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DRIVER_DRIVERCOMMAND_TD
#define SUPPORT_DRIVER_DRIVERCOMMAND_TD

include "llvm/Option/OptParser.td"

// A description of a command line executable, or subcommand of a top-level
// driver executable. In addition to documentation about the command, this also
// includes information about its positional input arguments.
class CommandDescription<string exec, string subcmd, string summ, string desc,
                         string inName, bit reqIn, bit varIn> {
  // The name of the executable.
  string executable = exec;
  // The name of a subcommand, or an empty string to indicate that a top-level
  // driver is being described.
  string subcommand = subcmd;
  // A one sentence summary. In manual pages, this is displayed next to the
  // command's name in the `NAME` section, as in:
  //
  //   mojo doc - Translate doc strings to a structured output format
  //
  // Note that here, the description 'executable' is set as "mojo", and the
  // 'subcommand' is set as "doc".
  //
  // The summary:
  // 1. Should begin with a capital letter.
  // 2. Should not end with a period.
  // 3. Should be in lowercase, except where English or technical terminological
  //    convention dictates otherwise.
  string summary = summ;
  // A long-form description of the command that can span multiple paragraphs.
  // This should begin with a capital letter, and end with a period.
  string description = desc;

  // The name to display as a placeholder for a positional input argument.
  // This should be a lowercase string.
  string inputMetaVarName = inName;
  // Whether this command requires an input argument. `true` indicates the
  // argument is required (the command fails if no input is provided), `false`
  // indicates it is optional (the command can succeed with no input argument,
  // or it uses a default input argument).
  bit requiresInput = reqIn;
  // Whether this command accepts more than one input argument.
  bit variadicInput = varIn;
}

// A description of a top-level driver executable.
class DriverDescription<
    string executable, string summary, string description,
    string inputMetaVarName, bit requiresInput, bit variadicInput> :
  CommandDescription<executable, /*subcommand=*/"", summary, description,
                     inputMetaVarName, requiresInput, variadicInput>;

// This mix-in record can be used to order option groups, options, and their
// aliases relative to one another.
// * Option groups are sorted in ascending index order.
// * Within each option group, options are sorted in ascending index order.
// * Among an option's aliases, aliases are sorted in ascending index order.
// * Wihtin any of these groupings, if two records have the same index, the
//   order is non-deterministic, and a warning is emitted.
// * Records without an index are placed in a non-deterministic order (based on
//   TableGen record ID) after records with an index. Warnings are emitted for
//   records without an index.
class Index<int i> {
  int index = i;
}

#endif // SUPPORT_DRIVER_DRIVERCOMMAND_TD
