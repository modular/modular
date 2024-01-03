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

// This multiclass defines two option records: one that accepts an `=` suffix,
// and one that doesn't. In other words, given the option name `foo` and the
// metavar name `FOO`, this defines both `--foo <FOO>` and `--foo=<FOO>`
multiclass CommandOption<list<string> prefixes, string name, OptionGroup group,
                         Index i, MetaVarName meta, HelpText help> {
  def NAME : Separate<prefixes, name>, Group<group>, Index<i.index>,
             MetaVarName<meta.MetaVarName>, HelpText<help.HelpText>;

  // The `=` alias is the first alias, at index 0. However, since it's hidden,
  // it will not occupy the first spot in the list of aliases.
  def NAME # _eq : Joined<prefixes, name # "=">, Group<group>,
                   Index<0>, Alias<!cast<Separate>(NAME)>, Flags<[HelpHidden]>;
}

// Like CommandOption, but also accepts Values.
multiclass CommandOptionWithValues<list<string> prefixes, string name,
                                   OptionGroup group, Index i, MetaVarName meta,
                                   HelpText help, Values values> {
  def NAME : Separate<prefixes, name>,
             Group<group>,
             Index<i.index>,
             MetaVarName<meta.MetaVarName>,
             HelpText<help.HelpText>,
             Values<values.Values>;

  // The `=` alias is the first alias, at index 0. However, since it's hidden,
  // it will not occupy the first spot in the list of aliases.
  def NAME #_eq : Joined<prefixes, name #"=">,
                  Group<group>,
                  Index<0>,
                  Alias<!cast<Separate>(NAME)>,
                  Flags<[HelpHidden]>;
}

// A collection of information related to a command's input, which is used to
// generate usage strings for a command.
class UsageInfo<string optName, string inName, bit varIn> {
  // The name to display for options, or an empty string if no options should be
  // displayed.
  string optionsName = optName;
  // The name to display as a placeholder for a positional input argument, or an
  // empty string if no positional input argument should be displayed. If
  // non-empty, this must be a lowercase string.
  string inputName = inName;
  // Whether to append a "..." to the positional input argument name.
  bit variadicInput = varIn;
}

// A description of a command line executable, or subcommand of a top-level
// driver executable.
class CommandDescription<string exec, string subcmd, string summ, string desc,
                         list<UsageInfo> uses> {
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
  // Information on how a command should be used. These are displayed in the
  // `SYNOPSIS` section in the order they are defined, as in:
  //
  //   mojo [options] <command>
  //   mojo [options] <path>
  //   mojo
  //
  // At least one should be present in this list.
  list<UsageInfo> usages = uses;
}

// A description of a top-level driver executable.
class DriverDescription<string executable, string summary, string description,
                        list<UsageInfo> usages> :
  CommandDescription<executable, /*subcommand=*/"", summary, description,
                     usages>;

#endif // SUPPORT_DRIVER_DRIVERCOMMAND_TD
