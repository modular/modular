//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Support/Resource.h"

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::AsyncRT;

/// If true, uses will be printed to llvm::errs() as they begin and end.
constexpr bool kTraceUsesToErrs = false;

//===----------------------------------------------------------------------===//
// Private utils
//===----------------------------------------------------------------------===//

/// Force llvm::errs() messages to be serialized.
static std::mutex &getMessageMutex() {
  static std::mutex messageMutex;
  return messageMutex;
}

static bool isReferencing(ResourceUseType type) {
  return type == kReferencingResourceUse;
}

static bool isReading(ResourceUseType type) {
  return type == kReadingResourceUse || type == kMutatingResourceUse;
}

static bool isWriting(ResourceUseType type) {
  return type == kWritingResourceUse || type == kMutatingResourceUse;
}

static const char *useTypeToString(ResourceUseType type) {
  switch (type) {
  case kInvalidResourceUse:
    return "invalid";
  case kReferencingResourceUse:
    return "referencing";
  case kReadingResourceUse:
    return "reading";
  case kWritingResourceUse:
    return "writing";
  case kMutatingResourceUse:
    return "mutating";
  }
  llvm_unreachable("invalid resource type");
}

static void printNames(const llvm::StringMap<size_t> &names,
                       llvm::raw_ostream &os) {
  llvm::interleaveComma(names, os,
                        [&os](const llvm::StringMapEntry<size_t> &entry) {
                          os << "'" << entry.first() << "'";
                          if (entry.second > 1)
                            os << "(" << entry.second << ")";
                        });
}

//===----------------------------------------------------------------------===//
// ResourceSection
//===----------------------------------------------------------------------===//

void M::AsyncRT::printResourceSection(llvm::raw_ostream &os,
                                      const ResourceSection &section) {
  if (isAllResourceSection(section)) {
    os << "all";
  } else {
    os << "[";
    os << section.start();
    os << ", ";
    os << section.end();
    os << ")";
  }
}

bool ResourceSections::containsSection(const ResourceSection &section) const {
  if (section.size() == 0)
    return false;
  return contains(section);
}

bool ResourceSections::overlapsSection(const ResourceSection &section) const {
  if (section.size() == 0)
    return false;
  return llvm::any_of(Ranges, [&section](const ResourceSection &existing) {
    return existing.intersects(section);
  });
}

void ResourceSections::addSection(const ResourceSection &section) {
  if (section.size() == 0)
    return;
  insert(section);
}

void ResourceSections::removeSection(const ResourceSection &section) {
  if (section.size() == 0)
    return;

  // Following uses direct pointers due to type checking glitch.
  const ResourceSection *itr = find(section.start(), section.end());
  assert(itr != Ranges.end() && "attempting to remove section not in sections");
  const ResourceSection *begin = Ranges.begin();
  size_t i = std::distance(begin, itr);

  ResourceSection newPrefix(itr->start(), section.start());
  ResourceSection newSuffix(section.end(), itr->end());
  if (newPrefix.size() && newSuffix.size()) {
    // Repurpose existing for prefix, insert suffix.
    Ranges[i] = newPrefix;
    insert(newSuffix);
  } else if (newPrefix.size()) {
    Ranges[i] = newPrefix;
  } else if (newSuffix.size()) {
    Ranges[i] = newSuffix;
  } else {
    // Remove entire existing section.
    Ranges.erase(itr);
  }
}

void ResourceSections::print(llvm::raw_ostream &os) const {
  for (const auto &section : Ranges) {
    os << "    ";
    printResourceSection(os, section);
    os << "\n";
  }
}

//===----------------------------------------------------------------------===//
// ResourceUse
//===----------------------------------------------------------------------===//

void ResourceUse::swap(ResourceUse &rhs) {
  std::swap(name, rhs.name);
  std::swap(resource, rhs.resource);
  std::swap(useType, rhs.useType);
  std::swap(section, rhs.section);
}

ResourceUse ResourceUse::copy() const {
  if (resource)
    return resource->beginUse(name, useType, section);
  else
    return {};
}

ResourceUse::~ResourceUse() {
  if (resource)
    resource->endUse(*this);
}

void ResourceUse::reset() {
  if (resource)
    resource->endUse(*this);
  name.clear();
  resource.reset();
  useType = kInvalidResourceUse;
  section = ResourceSection();
}

void ResourceUse::print(llvm::raw_ostream &os) const {
  os << "ResourceUse(";
  if (resource) {
    os << "'" << name << "'";
    os << " of '" << resource->name << "'";
    os << " for " << useTypeToString(useType);
    os << " of ";
    printResourceSection(os, section);
  } else {
    os << "null";
  }
  os << ")";
}

//===----------------------------------------------------------------------===//
// Resource
//===----------------------------------------------------------------------===//

Resource::~Resource() {
  // Though a dtor, we must still lock to guarantee we see the current state.
  std::lock_guard<std::mutex> guard(mu);

  if (state != kFreed)
    markFreedImpl();
}

RCRef<Resource> Resource::allocate(std::string name, bool isInitialized) {
#ifdef NDEBUG
  llvm::errs()
      << "CAUTION: Attempting to use AsyncRT::Resource on a build "
         "without asserts enabled. That's probably not what you intended.\n";
#endif
  return RCRef<Resource>::take(new Resource(std::move(name), isInitialized));
}

Resource::Resource(std::string name, bool isInitialized)
    : name(std::move(name)) {
  if (isInitialized)
    initialized.addSection(allResourceSection());
}

ResourceUse Resource::beginUse(std::string useName, ResourceUseType useType,
                               ResourceSection section) {
  ResourceUse use(std::move(useName), RCRef<Resource>::copy(this), useType,
                  section);

  if constexpr (kTraceUsesToErrs) {
    std::lock_guard<std::mutex> innerGuard(getMessageMutex());
    llvm::errs() << "begin ";
    use.print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs().flush();
  }

  std::lock_guard<std::mutex> guard(mu);

  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to use a freed resource", use);
  if (isReferencing(useType)) {
    if (auto errOr = checkForOverlappingSections(
            referencing, kContained,
            /*existingUseType=*/kReferencingResourceUse, use.useType,
            use.section))
      fatal(errOr.getError(), use);
  }
  if (isReading(useType)) {
    if (auto errOr = checkForOverlappingSections(
            reading, kEqual, /*existingUseType=*/kReadingResourceUse,
            use.useType, use.section))
      fatal(errOr.getError(), use);
    if (auto errOr = checkForOverlappingSections(
            writing, kExclusive, /*existingUseType=*/kWritingResourceUse,
            use.useType, use.section))
      fatal(errOr.getError(), use);
    if (!initialized.containsSection(use.section))
      fatal("attempting to read from uninitialized (section of) resource", use);
  }
  if (isWriting(useType)) {
    if (auto errOr = checkForOverlappingSections(
            reading, kExclusive, /*existingUseType=*/kReadingResourceUse,
            use.useType, use.section))
      fatal(errOr.getError(), use);
    if (auto errOr = checkForOverlappingSections(
            writing, kExclusive, /*existingUseType=*/kWritingResourceUse,
            use.useType, use.section))
      fatal(errOr.getError(), use);
  }

  // Make the state change.
  if (isReferencing(useType))
    addUseToMap(referencing, use.section, use.name);
  if (isReading(useType)) {
    addUseToMap(reading, use.section, use.name);
    allReading.addSection(use.section);
  }
  if (isWriting(useType)) {
    addUseToMap(writing, use.section, use.name);
    allWriting.addSection(use.section);
  }

  return use;
}

void Resource::markInitialized(const ResourceUse &use) {
  std::lock_guard<std::mutex> guard(mu);

  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to initialize an already freed resource");

  // Make the state change.
  initialized.addSection(use.section);
}

void Resource::markUninitialized(const ResourceUse &use) {
  std::lock_guard<std::mutex> guard(mu);

  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to uninitialize an already freed resource");
  if (allReading.overlapsSection(use.section))
    fatal("attempting to mark (section of) resource as uninitialized while it "
          "still has readers",
          use);

  // Make the state change.
  initialized.removeSection(use.section);
}

void Resource::markFreed() {
  std::lock_guard<std::mutex> guard(mu);
  markFreedImpl();
}

void Resource::endUse(const ResourceUse &use) {
  assert(use);

  if constexpr (kTraceUsesToErrs) {
    std::lock_guard<std::mutex> innerGuard(getMessageMutex());
    llvm::errs() << "end ";
    use.print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs().flush();
  }

  std::lock_guard<std::mutex> guard(mu);

  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to end use of a resource which has already been freed",
          use);

  // Make the state change.
  if (isReferencing(use.useType))
    (void)removeUseFromMap(referencing, use.section, use.name);
  if (isReading(use.useType)) {
    if (removeUseFromMap(reading, use.section, use.name))
      allReading.removeSection(use.section);
  }
  if (isWriting(use.useType)) {
    if (removeUseFromMap(writing, use.section, use.name))
      allWriting.removeSection(use.section);
  }
}

void Resource::markFreedImpl() {
  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to free an already freed resource");
  if (!referencing.empty())
    fatal("attempting to free a resource while it still has references");
  if (!allReading.empty())
    fatal("attempting to free a resource while it still has readers");
  if (!allWriting.empty())
    fatal("attempting to free a resource while it still has writers");

  // Make the state change.
  state = kFreed;
}

const char *Resource::stateToString(ResourceState state) {
  switch (state) {
  case kAlive:
    return "alive";
  case kFreed:
    return "freed";
  }
  llvm_unreachable("invalid resource state");
}

void Resource::print(llvm::raw_ostream &os) const {
  os << "Resource(\n";
  os << "  name: '" << name << "'\n";
  os << "  state: " << stateToString(state) << "\n";
  os << "  initialized sections:\n";
  initialized.print(os);
  os << "  sections being referenced:\n";
  for (const auto &[section, names] : referencing) {
    os << "    ";
    printResourceSection(os, section);
    os << " {";
    printNames(names, os);
    os << "}\n";
  }
  os << "  sections being read:\n";
  for (const auto &[section, names] : reading) {
    os << "    ";
    printResourceSection(os, section);
    os << " {";
    printNames(names, os);
    os << "}\n";
  }
  os << "  union being read:\n";
  allReading.print(os);
  os << "  sections being written:\n";
  for (const auto &[section, names] : writing) {
    os << "    ";
    printResourceSection(os, section);
    os << " {";
    printNames(names, os);
    os << "}\n";
  }
  os << "  union being written:\n";
  allWriting.print(os);
  os << ")\n";
}

void Resource::fatal(StringRef message, const ResourceUse &use) {
  std::lock_guard<std::mutex> guard(getMessageMutex());
  llvm::errs() << "invalid use of resource: " << message << "\n";
  llvm::errs() << "by use ";
  use.print(llvm::errs());
  llvm::errs() << "\n";
  print(llvm::errs());
  llvm::errs().flush();
  assert(false &&
         "invalid use of resource: see above error message for details");
}

void Resource::fatal(StringRef message) {
  std::lock_guard<std::mutex> guard(getMessageMutex());
  llvm::errs() << "invalid use of resource: " << message << "\n";
  print(llvm::errs());
  llvm::errs().flush();
  assert(false &&
         "invalid use of resource: see above error message for details");
}

void Resource::addUseToMap(UseMap &map, const ResourceSection &section,
                           StringRef useName) {
  ++map[section][useName];
}

bool Resource::removeUseFromMap(UseMap &map, const ResourceSection &section,
                                StringRef useName) {
  llvm::StringMap<size_t> &names = map[section];
  size_t &count = names[useName];
  assert(count > 0 && "unbalanced addUseToMap/removeUseFromMap calls");
  if (--count == 0)
    names.erase(useName);
  if (names.empty()) {
    map.erase(section);
    return true;
  }
  return false;
}

ErrorOrSuccess Resource::checkForOverlappingSections(
    const UseMap &map, UsageRule usageRule, ResourceUseType existingUseType,
    ResourceUseType desiredUseType, const ResourceSection &section) {
  for (auto &[existingSection, names] : map) {
    if (existingSection.intersects(section)) {
      switch (usageRule) {
      case kContained:
        // Ok provided section is a (possibly equal) sub-section of existing.
        if (existingSection.contains(section))
          continue;
        break;
      case kEqual:
        // Ok provided equal.
        if (existingSection == section)
          continue;
        break;
      case kExclusive:
        // No overlap is allowed.
        break;
      }

      std::string str;
      llvm::raw_string_ostream os(str);
      os << "requested section for " << useTypeToString(desiredUseType)
         << " overlaps with existing section ";
      printResourceSection(os, existingSection);
      os << " for " << useTypeToString(existingUseType)
         << " with active uses {";
      printNames(names, os);
      os << "}";
      return Error(str);
    }
  }
  return success();
}
