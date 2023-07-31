//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/Resource.h"

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;
using namespace M::LLCL;

//===----------------------------------------------------------------------===//
// Private utils
//===----------------------------------------------------------------------===//

static bool isAllSection(const ResourceSection &section) {
  return section.start() == kMinResourceOffset &&
         section.end() == kMaxResourceOffset;
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
}

/// Returns true if section overlaps without being exactly equal to
/// any key in map.
static bool
hasOverlappingKey(const llvm::DenseMap<ResourceSection, size_t> &map,
                  const ResourceSection &section) {
  return llvm::any_of(
      map, [&section](const std::pair<ResourceSection, size_t> &pair) {
        return pair.first.intersects(section) && pair.first != section;
      });
}

//===----------------------------------------------------------------------===//
// ResourceSection
//===----------------------------------------------------------------------===//

void M::LLCL::printResourceSection(llvm::raw_ostream &os,
                                   const ResourceSection &section) {
  if (isAllSection(section)) {
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
};

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
      << "CAUTION: Attempting to use LLCL::Resource on a build "
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
  std::lock_guard<std::mutex> guard(mu);

  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to use a freed resource", use);
  if (isReferencing(useType)) {
    if (hasOverlappingKey(numReferencing, use.section))
      fatal("attempting to reference overlapping sections of resource", use);
  }
  if (isReading(useType)) {
    if (hasOverlappingKey(numReading, use.section))
      fatal("attempting to read from overlapping sections of resource", use);
    if (allWriting.overlapsSection(use.section))
      fatal("attempting to read from (section of) resource which is also "
            "being written",
            use);
    if (!initialized.containsSection(use.section))
      fatal("attempting to read from uninitialized (section of) resource", use);
  }
  if (isWriting(useType)) {
    if (allReading.overlapsSection(use.section))
      fatal("attempting to write to (section of) resource which is also being "
            "read",
            use);
    if (allWriting.overlapsSection(use.section))
      fatal("attempting to write to (section of) resource which is also being "
            "written",
            use);
  }

  // Make the state change.
  addUse(use.name);
  if (isReferencing(useType))
    addReferencing(use.section);
  if (isReading(useType)) {
    addReading(use.section);
    allReading.addSection(use.section);
  }
  if (isWriting(useType))
    allWriting.addSection(use.section);

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
  std::lock_guard<std::mutex> guard(mu);

  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to end use of a resource which has already been freed",
          use);

  // Make the state change.
  removeUse(use.name);
  if (isReferencing(use.useType))
    removeReferencing(use.section);
  if (isReading(use.useType)) {
    if (removeReading(use.section))
      allReading.removeSection(use.section);
  }
  if (isWriting(use.useType))
    allWriting.removeSection(use.section);
}

void Resource::markFreedImpl() {
  // Run the gauntlet of tests.
  if (state == kFreed)
    fatal("attempting to free an already freed resource");
  if (!numReferencing.empty())
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
}

void Resource::print(llvm::raw_ostream &os) const {
  os << "Resource(\n";
  os << "  name: '" << name << "'\n";
  os << "  state: " << stateToString(state) << "\n";
  os << "  initialized sections:\n";
  initialized.print(os);
  os << "  in use by:\n";
  for (const auto &pair : uses) {
    if (pair.second)
      os << "     '" << pair.first() << "' (" << pair.second << ")\n";
  }
  os << "  referencing by section:\n";
  for (const auto &pair : numReferencing) {
    os << "    ";
    printResourceSection(os, pair.first);
    os << " (" << pair.second << ")\n";
  }
  os << "  sections being read:\n";
  allReading.print(os);
  os << "  reading by section:\n";
  for (const auto &pair : numReading) {
    os << "    ";
    printResourceSection(os, pair.first);
    os << " (" << pair.second << ")\n";
  }
  os << "  sections being written:\n";
  allWriting.print(os);
  os << ")\n";
}

void Resource::fatal(StringRef message, const ResourceUse &use) {
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
  llvm::errs() << "invalid use of resource: " << message << "\n";
  print(llvm::errs());
  llvm::errs().flush();
  assert(false &&
         "invalid use of resource: see above error message for details");
}

void Resource::addUse(StringRef useName) { ++uses[useName]; }

void Resource::removeUse(StringRef useName) {
  size_t &n = uses[useName];
  assert(n > 0 && "unbalanced addUse/removeUse calls");
  if (--n == 0)
    uses.erase(useName);
}

void Resource::addReferencing(const ResourceSection &section) {
  ++numReferencing[section];
}

void Resource::removeReferencing(const ResourceSection &section) {
  size_t &n = numReferencing[section];
  assert(n > 0 && "unbalanced addReferencing/removeReferencing calls");
  if (--n == 0)
    numReferencing.erase(section);
}

void Resource::addReading(const ResourceSection &section) {
  ++numReading[section];
}

bool Resource::removeReading(const ResourceSection &section) {
  size_t &n = numReading[section];
  assert(n > 0 && "unbalanced addReading/removeReading calls");
  if (--n > 0)
    return false;
  numReading.erase(section);
  return true;
}
