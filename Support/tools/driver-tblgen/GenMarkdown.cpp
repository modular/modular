//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a TableGen backend that, given a command description and
// option groups, outputs Markdown that can be embedded into a static website
// generator such as Quarto.
//
//===----------------------------------------------------------------------===//

#include "GenMarkdown.h"
#include "BackendRegistry.h"
#include "DriverCommand.h"

#include "Support/LLVMForwardDecls.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace M;

static void genTitle(raw_ostream &os, const CommandDescription &cmd) {
  os << "# " << cmd.getName(/*join=*/" ") << "\n\n";
}

static void genNameSection(raw_ostream &os, const CommandDescription &cmd) {
  os << "## Name\n\n"
     << llvm::formatv("{0} – {1}\n\n", cmd.getName(), cmd.getSummary());
}

static void genSynopsisSection(raw_ostream &os, const CommandDescription &cmd,
                               ArrayRef<CommandOptionGroup> groups) {
  os << "## Synopsis\n\n"
     << "```\n"
     << cmd.getName(/*join=*/" ");
  if (!groups.empty())
    os << " [options]";
  std::string input = llvm::formatv("{0}{1}", cmd.getInputMetaVarName(),
                                    cmd.getVariadicInput() ? "..." : "");
  if (!cmd.getRequiresInput())
    input = "[" + input + "]";
  os << ' ' << input << "\n"
     << "```\n\n";
}

static void genDescriptionSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  os << "## Description\n\n" << cmd.getDescription() << "\n\n";
}

/// Output the given LLVM `Option` record's prefix and name, followed by its
/// `MetaVarName` if present.
static void genOptionName(raw_ostream &os, const llvm::Record *option) {
  os << CommandOption::getPreferredPrefix(option)
     << option->getValueAsString("Name");

  if (auto metaVarName = option->getValueAsOptionalString("MetaVarName")) {
    if (option->getValueAsDef("Kind")->getValueAsString("Name") != "Joined")
      os << ' ';
    os << '<' << *metaVarName << '>';
  }
}

/// If there are 1 or more option groups present, outputs an "OPTIONS" section,
/// with a separate sub-section for each option group.
static void genOptionsSection(raw_ostream &os,
                              ArrayRef<CommandOptionGroup> groups) {
  if (groups.empty())
    return;

  os << "## Options\n\n";

  for (const CommandOptionGroup &group : groups) {
    // Print each option group, and its help text if available.
    os << "### " << group.getGroup()->getValueAsString("Name") << "\n\n";
    if (std::optional<StringRef> helpText =
            group.getGroup()->getValueAsOptionalString("HelpText"))
      os << *helpText << "\n\n";

    // Print all the options that belong to this group.
    for (const CommandOption &option : group.getOptions()) {
      // Print the option's name, and then the names of its aliases.
      os << "#### `";
      genOptionName(os, option.getOption());
      os << '`';
      for (const llvm::Record *option : option.getAliases()) {
        os << ", `";
        genOptionName(os, option);
        os << '`';
      }
      os << "\n\n";

      // Print the main option's help text (the aliases' help text is ignored).
      // The help text may be an empty string, if the documentation writer
      // ignored mojo-tblgen warnings.
      os << option.getHelpText() << "\n\n";
    }
  }
}

static bool genHelpText(raw_ostream &os, const llvm::RecordKeeper &records) {
  ErrorOr<CommandDescription> cmd = CommandDescription::get(records);
  if (cmd.isError()) {
    llvm::PrintError(cmd.getError());
    return true;
  }
  std::vector<CommandOptionGroup> groups = CommandOptionGroup::getAll(records);

  genTitle(os, *cmd);
  genNameSection(os, *cmd);
  genSynopsisSection(os, *cmd, groups);
  genDescriptionSection(os, *cmd);
  genOptionsSection(os, groups);
  return false;
}

void M::registerGenMarkdownBackend(BackendRegistry &registry) {
  registry.addBackend("gen-markdown", "Generate help text as Markdown",
                      [](raw_ostream &os, const llvm::RecordKeeper &records) {
                        return genHelpText(os, records);
                      });
}
