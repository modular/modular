//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines a TableGen backend that, given a command description and
// option groups, outputs a raw C++ string literal that can be used as help
// text.
//
//===----------------------------------------------------------------------===//

#include "GenHelpText.h"
#include "BackendRegistry.h"
#include "DriverCommand.h"

#include "Support/LLVMForwardDecls.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace M;

/// Write the given `text` to the output stream `os`, inserting a line break if
/// a "word" would exceed the column `limit` (this is a simple function that,
/// for now, treats spaces as word delimiters. It will need to be updated if we
/// wish to better support splitting `inline code` or hyphenated-text). Each new
/// line is indented by `indent`.
static raw_ostream &writeWordWrapped(raw_ostream &os, Twine text,
                                     size_t indent = 0, size_t limit = 80) {
  unsigned maxLineLength = limit - indent;

  SmallVector<char> buffer;
  StringRef str = text.toStringRef(buffer);

  // Write the first word, indented.
  auto [word, rest] = str.split(' ');
  os.indent(indent) << word;
  size_t remainingLength = maxLineLength - word.size();
  str = rest;
  // Write all remaining words.
  while (!str.empty()) {
    auto [word, rest] = str.split(' ');
    if (remainingLength < word.size() + 1) {
      // Not enough space to write a word and a space; write a new line,
      // re-indent, and reset the remaining length.
      os << '\n';
      os.indent(indent) << word;
      remainingLength = maxLineLength - word.size();
    } else {
      // Enough space to write a space and and the next word. Subtract what we
      // wrote from the remaining length.
      os << ' ' << word;
      remainingLength -= word.size() + 1;
    }
    str = rest;
  }

  return os;
}

static void genNameSection(raw_ostream &os, const CommandDescription &cmd) {
  os << "NAME\n";
  writeWordWrapped(os,
                   llvm::formatv("{0} — {1}", cmd.getName(), cmd.getSummary()),
                   /*indent=*/8)
      << "\n\n";
}

static void genSynopsisSection(raw_ostream &os, const CommandDescription &cmd,
                               ArrayRef<CommandOptionGroup> groups) {
  os << "SYNOPSIS\n";
  os.indent(8) << cmd.getName(/*join=*/" ");
  if (!groups.empty())
    os << " [options]";
  std::string input = llvm::formatv("{0}{1}", cmd.getInputMetaVarName(),
                                    cmd.getVariadicInput() ? "..." : "");
  if (!cmd.getRequiresInput())
    input = "[" + input + "]";
  os << ' ' << input << "\n\n";
}

static void genDescriptionSection(raw_ostream &os,
                                  const CommandDescription &cmd) {
  os << "DESCRIPTION\n";
  writeWordWrapped(os, cmd.getDescription(), /*indent=*/8);
  os << "\n\n";
}

/// Output the given LLVM `Option` record's prefix and name, followed by its
/// `MetaVarName` if present.
static void genOptionName(raw_ostream &os, const llvm::Record *option,
                          size_t indent = 0) {
  os.indent(indent) << CommandOption::getPreferredPrefix(option)
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

  os << "OPTIONS\n";

  for (const CommandOptionGroup &group : groups) {
    // Print each option group, and its help text if available.
    os.indent(4) << group.getGroup()->getValueAsString("Name") << '\n';
    if (std::optional<StringRef> helpText =
            group.getGroup()->getValueAsOptionalString("HelpText"))
      writeWordWrapped(os, *helpText, /*indent=*/8) << "\n\n";

    // Print all the options that belong to this group.
    for (const CommandOption &option : group.getOptions()) {
      // Print the option's name, and then the names of its aliases.
      genOptionName(os, option.getOption(), /*indent=*/8);
      for (const llvm::Record *option : option.getAliases()) {
        os << ", ";
        genOptionName(os, option);
      }
      os << '\n';

      // Print the main option's help text (the aliases' help text is ignored).
      // The help text may be an empty string, if the documentation writer
      // ignored mojo-tblgen warnings.
      writeWordWrapped(os, option.getHelpText(), /*indent=*/12) << "\n\n";
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

  os << "u8R\"(";
  genNameSection(os, *cmd);
  genSynopsisSection(os, *cmd, groups);
  genDescriptionSection(os, *cmd);
  genOptionsSection(os, groups);
  os << ")\"";
  return false;
}

void M::registerGenHelpTextBackend(BackendRegistry &registry) {
  registry.addBackend("gen-help-text",
                      "Generate help text as a C++ constant string",
                      [](raw_ostream &os, const llvm::RecordKeeper &records) {
                        return genHelpText(os, records);
                      });
}
