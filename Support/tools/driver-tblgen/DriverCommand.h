//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TOOLS_DRIVERTBLGEN_DRIVERCOMMAND_H
#define SUPPORT_TOOLS_DRIVERTBLGEN_DRIVERCOMMAND_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include <cassert>
#include <string>

namespace M {

class CommandOption;

/// A convenience wrapper for a `CommandDescription` TableGen record. This
/// provides getters for the record's values, as well as other helper functions.
///
/// Instead of constructing instances of this class directly, use the static
/// `get` member function to construct one based on parsed TableGen records.
class CommandDescription {
public:
  /// Given a set of parsed TableGen records, return either a concrete command
  /// description, or an error if none could be found. This also emits warning
  /// diagnostics if more than one command description is found.
  static ErrorOr<CommandDescription> get(const llvm::RecordKeeper &records);

  StringRef getExecutable() const {
    return record->getValueAsString("executable");
  }

  StringRef getSubcommand() const {
    return record->getValueAsString("subcommand");
  }

  StringRef getSummary() const { return record->getValueAsString("summary"); }

  StringRef getDescription() const {
    return record->getValueAsString("description");
  }

  StringRef getInputMetaVarName() const {
    return record->getValueAsString("inputMetaVarName");
  }

  bool getRequiresInput() const {
    return record->getValueAsBit("requiresInput");
  }

  bool getVariadicInput() const {
    return record->getValueAsBit("variadicInput");
  }

  /// Given an joining string, joins the command description record's executable
  /// and subcommand values by that string.
  std::string getName(Twine join = "-") const {
    return llvm::formatv("{0}{1}{2}", getExecutable(),
                         getSubcommand().empty() ? "" : join, getSubcommand());
  }

private:
  /// Initializes the wrapper with the given `CommandDescription` record.
  CommandDescription(const llvm::Record *record) : record(record) {
    assert(record->isSubClassOf("CommandDescription") &&
           "unexpected record class");
  }

  const llvm::Record *record;
};

/// A wrapper around an LLVM `OptionGroup` record, as well as all of the
/// (continuously sorted) options that belong to that group. This helps backends
/// print options group-wise.
///
/// Instead of constructing instances of this class directly, use the static
/// `getAll` member function to construct a collection of them based on parsed
/// TableGen records.
class CommandOptionGroup {
public:
  /// Given a set of parsed TableGen records, returns a sorted list of all the
  /// option groups defined therein, along with their options.
  static std::vector<CommandOptionGroup>
  getAll(const llvm::RecordKeeper &records);

  /// Return the underlying LLVM `OptionGroup` record.
  const llvm::Record *getGroup() const { return group; }
  /// Return all the options that belong to this group.
  ArrayRef<CommandOption> getOptions() const { return options; }

  StringRef getGroupName() const {
    return getGroup()->getValueAsString("Name");
  }

  /// Return the option group's index value, if one is defined.
  std::optional<int64_t> getIndex() const;

  /// Given an LLVM `Option` record, either add it to the sorted list of group
  /// options, or return the option that was already added.
  CommandOption &findOrCreateOption(const llvm::Record *option);

private:
  /// Initializes the wrapper with the given `OptionGroup` record.
  CommandOptionGroup(const llvm::Record *group) : group(group) {
    assert(group->isSubClassOf("OptionGroup") && "unexpected record class");
  }

  const llvm::Record *group;
  std::vector<CommandOption> options;
};

/// A wrapper around an LLVM `Option` record, plus all of its aliases, which are
/// stored in a continuously sorted list. This helps backends print options and
/// their aliases side-by-side.
///
/// Instead of constructing instances of this class directly, use the
/// `CommandOptionGroup::getAll` member function to construct a collection of
/// groups and their options, based on parsed TableGen records.
class CommandOption {
public:
  /// Return the underlying LLVM `Option` record.
  const llvm::Record *getOption() const { return option; }

  /// Whether this option is a flag, meaning an option that takes no values.
  bool isFlag() const {
    return option->getValueAsDef("Kind")->getValueAsString("Name") == "Flag";
  }

  /// Return the option's help text, or an empty string if none exists.
  StringRef getHelpText() const {
    auto helpText = option->getValueAsOptionalString("HelpText");
    return helpText ? *helpText : "";
  }

  /// Return the option's optional metavar name, if one is defined.
  std::optional<StringRef> getMetaVarName() const {
    return option->getValueAsOptionalString("MetaVarName");
  }

  /// Return the option's index value, if one is defined.
  std::optional<int64_t> getIndex() const;

  /// Add an LLVM `Option` to the sorted list of aliases for this option.
  void addAlias(const llvm::Record *alias);

  /// Return all the aliases of this option.
  ArrayRef<const llvm::Record *> getAliases() const { return aliases; }

  /// Return the first prefix defined for the given `option`, which we treat as
  /// the "preferred" prefix for help text.
  static StringRef getPreferredPrefix(const llvm::Record *option) {
    std::vector<StringRef> prefixes =
        option->getValueAsListOfStrings("Prefixes");
    // Only options such as `INPUT` and `UNKNOWN` can be defined without a
    // prefix, and we don't process those.
    assert(!prefixes.empty() && "all options must have a prefix");
    return prefixes.front();
  }

private:
  /// Initializes the wrapper with the given `Option` record.
  CommandOption(const llvm::Record *option) : option(option) {
    assert(option->isSubClassOf("Option") && "unexpected record class");
  }
  /// Allow `CommandOptionGroup` to construct instances of this class.
  friend CommandOption &
  CommandOptionGroup::findOrCreateOption(const llvm::Record *);

  const llvm::Record *option;
  SmallVector<const llvm::Record *> aliases;
};

/// A comparator that can be used to sort option groups and options, based on
/// their index, in ascending order.
struct LessIndex {
  bool operator()(const llvm::Record *lhs, const llvm::Record *rhs) const;

  bool operator()(const CommandOptionGroup &lhs,
                  const CommandOptionGroup &rhs) const {
    return operator()(lhs.getGroup(), rhs.getGroup());
  }

  bool operator()(const CommandOption &lhs, const CommandOption &rhs) const {
    return operator()(lhs.getOption(), rhs.getOption());
  }
};

} // namespace M

#endif // SUPPORT_TOOLS_DRIVERTBLGEN_DRIVERCOMMAND_H
