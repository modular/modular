//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "DriverCommand.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include <algorithm>
#include <cassert>

using namespace M;

/// If the given text is non-empty, prints a warning if it does not begin with a
/// lowercase character, and returns false. If it is empty, prints a warning and
/// returns true.
static bool validateCapitalized(StringRef text, ArrayRef<llvm::SMLoc> locs,
                                Twine description) {
  if (text.empty()) {
    llvm::PrintWarning(locs, description + " should not be empty");
    return true;
  }

  if (!isupper(text.front()))
    llvm::PrintWarning(locs,
                       description + " should begin with a capital letter");

  return false;
}

/// Given a TableGen record, return its 'index' integer value, if one is
/// defined.
static std::optional<int64_t>
getValueAsOptionalIndex(const llvm::Record *record) {
  const llvm::RecordVal *val = record->getValue("index");
  if (!val)
    return {};

  llvm::IntInit *i = dyn_cast_if_present<llvm::IntInit>(val->getValue());
  if (!i)
    return {};

  return i->getValue();
}

//===----------------------------------------------------------------------===//
// CommandDescription
//===----------------------------------------------------------------------===//

ErrorOr<CommandDescription>
M::CommandDescription::get(const llvm::RecordKeeper &records) {
  std::vector<llvm::Record *> descriptions =
      records.getAllDerivedDefinitions("CommandDescription");
  if (descriptions.empty())
    return Error("you must define a 'CommandDescription' record");

  const llvm::Record *topLevel = nullptr;
  std::vector<const llvm::Record *> subcommands;
  for (const llvm::Record *record : descriptions) {
    if (record->getValueAsString("subcommand").empty()) {
      if (topLevel) {
        llvm::PrintError(
            record->getLoc(),
            "a second top-level 'CommandDescription' record is defined here");
        llvm::PrintNote(topLevel->getLoc(),
                        "a top-level 'CommandDescription' record has "
                        "already been defined here");
        return Error(
            "cannot define a second top-level 'CommandDescription' record");
      }
      topLevel = record;
    } else {
      subcommands.push_back(record);
    }
  }
  if (!topLevel && subcommands.size() > 1) {
    llvm::PrintError(
        subcommands.back()->getLoc(),
        "a second subcommand 'CommandDescription' is defined here");
    llvm::PrintNote(
        subcommands.front()->getLoc(),
        "a subcommand 'CommandDescription' has already been defined here");
    return Error("cannot define a second subcommand 'CommandDescription'");
  }

  const llvm::Record *record = nullptr;
  if (topLevel) {
    record = topLevel;
  } else {
    record = subcommands.back();
    subcommands.clear();
  }

  // Now that we have our description and its subcommands, perform some
  // validation.
  auto validateDescription = [](const llvm::Record *record) {
    if (record->getValueAsString("executable").empty())
      llvm::PrintWarning(record->getLoc(),
                         "command executable name should not be empty");

    StringRef summary = record->getValueAsString("summary");
    validateCapitalized(summary, record->getLoc(), "command summary");
    if (summary.ends_with("."))
      llvm::PrintWarning(record->getLoc(),
                         "command summary should not end with a period");

    StringRef description = record->getValueAsString("description");
    validateCapitalized(description, record->getLoc(), "command description");
    if (!description.ends_with("."))
      llvm::PrintWarning(record->getLoc(),
                         "command description should end with a period");

    std::vector<llvm::Record *> usages = record->getValueAsListOfDefs("usages");
    if (usages.empty())
      llvm::PrintWarning(record->getLoc(),
                         "command should include at least one usage");
    for (const llvm::Record *usage : usages) {
      StringRef optionsName = usage->getValueAsString("optionsName");
      if (optionsName.lower() != optionsName)
        llvm::PrintWarning(usage->getLoc(),
                           "usage options name should be lowercase");

      StringRef metaVarName = usage->getValueAsString("inputName");
      if (metaVarName.lower() != metaVarName)
        llvm::PrintWarning(usage->getLoc(),
                           "usage input metavar name should be lowercase");
    }
  };

  // First validate the main description.
  validateDescription(record);

  // Then, validate any subcommands it may have.
  DenseMap<int64_t, const llvm::Record *> indices;
  for (const llvm::Record *sub : subcommands) {
    validateDescription(sub);

    // In addition to the standard validations applied to all descriptions,
    // subcommands must also be ordered by index.
    std::optional<int64_t> index = getValueAsOptionalIndex(sub);
    if (!index) {
      llvm::PrintWarning(
          sub->getLoc(),
          llvm::formatv(
              "subcommand '{0}' has no index with which to order it by; "
              "it will appear in a non-deterministic order",
              sub->getValueAsString("subcommand")));
      continue;
    }

    if (!indices.insert({*index, sub}).second) {
      llvm::PrintWarning(
          sub->getLoc(),
          llvm::formatv(
              "subcommand '{0}' has index {1}, which has already been "
              "used; it will appear in a non-deterministic order",
              sub->getValueAsString("subcommand"), *index));
      const llvm::Record *previous = indices[*index];
      llvm::PrintNote(
          previous->getLoc(),
          llvm::formatv(
              "subcommand '{0}' has already been defined with index {1} here",
              previous->getValueAsString("subcommand"), *index));
    }
  }

  llvm::sort(subcommands, LessIndex());
  return CommandDescription(record, subcommands);
}

//===----------------------------------------------------------------------===//
// CommandOptionGroup
//===----------------------------------------------------------------------===//

std::vector<CommandOptionGroup>
M::CommandOptionGroup::getAll(const llvm::RecordKeeper &records) {
  // Create a sorted list of groups.
  std::vector<CommandOptionGroup> groups;
  std::vector<llvm::Record *> groupRecords =
      records.getAllDerivedDefinitions("OptionGroup");
  groups.reserve(groupRecords.size());
  llvm::transform(
      groupRecords, std::back_inserter(groups),
      [](const llvm::Record *record) { return CommandOptionGroup(record); });
  llvm::sort(groups, LessIndex());

  // For each group, add the options that belong to that group.
  // First, bucket all option records based on their group record.
  DenseMap<const llvm::Record *, std::vector<const llvm::Record *>>
      groupOptions;
  llvm::SmallSet<int64_t, 4> groupIndices;
  for (llvm::Record *option : records.getAllDerivedDefinitions("Option"))
    if (llvm::Record *group = option->getValueAsOptionalDef("Group"))
      groupOptions[group].push_back(option);
  // Then, add each group record's options to their (sorted) lists.
  for (CommandOptionGroup &group : groups) {
    llvm::SmallSet<int64_t, 4> optionIndices;
    for (const llvm::Record *option : groupOptions[group.getGroup()]) {
      // If the option is an alias, don't add it to the group, add it to its
      // aliased option.
      if (llvm::Record *aliased = option->getValueAsOptionalDef("Alias")) {
        CommandOption &aliasedOption = group.findOrCreateOption(aliased);
        aliasedOption.addAlias(option);
        continue;
      }

      group.findOrCreateOption(option);
    }

    // Now that we've constructed a group and all of its options, perform some
    // additional validation.
    if (std::optional<int64_t> index = group.getIndex()) {
      if (!groupIndices.insert(*index).second) {
        llvm::PrintWarning(
            group.getGroup()->getLoc(),
            llvm::formatv("group '{0}' has index {1}, which has already been "
                          "used; it will appear in a non-deterministic order",
                          group.getGroupName(), *index));
      }
    } else {
      llvm::PrintWarning(
          group.getGroup()->getLoc(),
          llvm::formatv("group '{0}' has no index with which to order it by; "
                        "it will appear in a non-deterministic order",
                        group.getGroupName()));
    }

    if (group.getOptions().empty())
      llvm::PrintWarning(
          group.getGroup()->getLoc(),
          llvm::formatv("publicly documented group '{0}' has no publicly "
                        "documented options",
                        group.getGroupName()));
  }

  return groups;
}

std::optional<int64_t> CommandOptionGroup::getIndex() const {
  return getValueAsOptionalIndex(group);
}

CommandOption &
M::CommandOptionGroup::findOrCreateOption(const llvm::Record *option) {
  assert(group == option->getValueAsDef("Group") &&
         "option does not belong to this group");

  auto it = llvm::lower_bound(options, CommandOption(option), LessIndex());
  if (it != options.end() && it->getOption() == option)
    return *it;

  CommandOption &result = *options.insert(it, CommandOption(option));

  // Now that we're processing this option for the first time, perform some
  // validation.
  StringRef name = option->getValueAsString("Name");
  StringRef helpText = result.getHelpText();
  validateCapitalized(
      helpText, option->getLoc(),
      llvm::formatv("help text for publicly visible option '{0}'", name));

  if (std::optional<StringRef> metaVarName = result.getMetaVarName()) {
    if (metaVarName->empty())
      llvm::PrintWarning(
          option->getLoc(),
          llvm::formatv("option '{0}' metavar name should not be empty", name));
    if (metaVarName->upper() != *metaVarName)
      llvm::PrintWarning(
          option->getLoc(),
          llvm::formatv("option '{0}' metavar name should be uppercase", name));
  } else if (!result.isFlag()) {
    llvm::PrintWarning(option->getLoc(),
                       llvm::formatv("option '{0}' takes a value, but does not "
                                     "define a metavar name for that value",
                                     name));
  }

  if (!result.getIndex())
    llvm::PrintWarning(
        option->getLoc(),
        llvm::formatv("option '{0}' has no index with which to order it by; "
                      "it will appear in a non-deterministic order",
                      name));

  return result;
}

//===----------------------------------------------------------------------===//
// CommandOption
//===----------------------------------------------------------------------===//

std::optional<int64_t> CommandOption::getIndex() const {
  return getValueAsOptionalIndex(option);
}

void CommandOption::addAlias(const llvm::Record *alias) {
  assert(alias->isSubClassOf("Option") && "unexpected record class");

  std::optional<int64_t> aliasIndex = getValueAsOptionalIndex(alias);
  auto it = llvm::lower_bound(aliases, alias, LessIndex());
  if (it != aliases.end()) {
    // If the alias already exists in the collection, no need to insert it.
    if (*it == alias)
      return;

    // If we're inserting an alias behind another, they may have the same
    // index value. If so, emit a warning.
    if (std::optional<int64_t> index = getValueAsOptionalIndex(*it))
      if (aliasIndex && aliasIndex == *index)
        llvm::PrintWarning(
            alias->getLoc(),
            llvm::formatv("alias '{0}' has index {1}, which has already been "
                          "used; it will appear in a non-deterministic order",
                          alias->getValueAsString("Name"), *aliasIndex));
  }

  // Now that we're adding this alias for the first time, perform some
  // validation.
  if (!aliasIndex)
    llvm::PrintWarning(
        alias->getLoc(),
        llvm::formatv("alias '{0}' has no index with which to order it by; "
                      "it will appear in a non-deterministic order",
                      alias->getValueAsString("Name")));

  aliases.insert(it, alias);
}

//===----------------------------------------------------------------------===//
// LessIndex
//===----------------------------------------------------------------------===//

bool LessIndex::operator()(const llvm::Record *lhs,
                           const llvm::Record *rhs) const {
  if (std::optional<int64_t> lhsIndex = getValueAsOptionalIndex(lhs))
    if (std::optional<int64_t> rhsIndex = getValueAsOptionalIndex(rhs))
      return lhsIndex < rhsIndex;
  return llvm::LessRecordByID()(lhs, rhs);
}
