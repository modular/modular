//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Bookkeeping for the runtime detection of use-after-free, use-of-uninit and
// data race bugs for abstract 'uses' of 'resources'. Only practical in debug
// builds.
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_RESOURCE_H
#define LLCL_SUPPORT_RESOURCE_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>

namespace M::AsyncRT {

//===----------------------------------------------------------------------===//
// ResourceSection
//===----------------------------------------------------------------------===//

/// A 'section' of a resource. Has no specific interpretation by the resource
/// checking machinery beyond its [min, max] integer range. The 'all' section
/// can be used to represent the entire resource.
using ResourceSection = llvm::AddressRange;

constexpr uint64_t kMinResourceOffset = 0;
constexpr uint64_t kMaxResourceOffset =
    std::numeric_limits<uint64_t>::max() - 3;

// Two reserved values are needed by DenseMap. We avoid max to make the
// hash function less fussy.
constexpr uint64_t kReservedResourceOffset1 =
    std::numeric_limits<uint64_t>::max() - 2;
constexpr uint64_t kReservedResourceOffset2 =
    std::numeric_limits<uint64_t>::max() - 1;

/// Returns the 'all' section.
inline ResourceSection allResourceSection() {
  return ResourceSection(kMinResourceOffset, kMaxResourceOffset);
}

/// Returns true if section is the 'all' section.
inline bool isAllResourceSection(const ResourceSection &section) {
  return section.start() == kMinResourceOffset &&
         section.end() == kMaxResourceOffset;
}

void printResourceSection(llvm::raw_ostream &os,
                          const ResourceSection &section);

} // namespace M::AsyncRT

namespace llvm {
/// Allow ResourceSection to be used as a DenseMap key.
template <>
struct DenseMapInfo<M::AsyncRT::ResourceSection> {
  static inline M::AsyncRT::ResourceSection getEmptyKey() {
    return M::AsyncRT::ResourceSection(M::AsyncRT::kReservedResourceOffset1,
                                       M::AsyncRT::kReservedResourceOffset1);
  }
  static inline M::AsyncRT::ResourceSection getTombstoneKey() {
    return M::AsyncRT::ResourceSection(M::AsyncRT::kReservedResourceOffset2,
                                       M::AsyncRT::kReservedResourceOffset2);
  }
  static unsigned getHashValue(const M::AsyncRT::ResourceSection &section) {
    // Mix the start and end values, avoiding collapse due to zeros.
    uint64_t h = (section.start() + 1) * (section.end() + 1);
    // Fold back to 32 bits.
    static_assert(sizeof(unsigned) == 4);
    return static_cast<unsigned>(h >> 32) ^ static_cast<unsigned>(h);
  }
  static bool isEqual(const M::AsyncRT::ResourceSection &lhs,
                      const M::AsyncRT::ResourceSection &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace M::AsyncRT {

/// A set of resource sections.
class ResourceSections : private llvm::AddressRanges {
public:
  using llvm::AddressRanges::AddressRanges;
  using llvm::AddressRanges::empty;

  /// Returns true if sections contains all of section.
  bool containsSection(const ResourceSection &section) const;
  /// Returns true if sections overlaps section.
  bool overlapsSection(const ResourceSection &section) const;
  /// Adds section to sections in-place.
  void addSection(const ResourceSection &section);
  /// Removes section from sections in-place.
  void removeSection(const ResourceSection &section);
  void print(llvm::raw_ostream &os) const;
};

//===----------------------------------------------------------------------===//
// ResourceUse
//===----------------------------------------------------------------------===//

/// How a resource is 'used'.
enum ResourceUseType {
  kInvalidResourceUse = 0,
  // The use is not (yet) reading or writing, but we still wish to detect
  // use-after-free errors.
  //
  // For example, GML BufferRefs may be shared between many operations, however
  // at any one time only some operations will be in flight, and those in-flight
  // operations may be for reading or writing. In this case a
  // kReferencingResourceUse can be used to represent the longer lifetime of
  // the BufferRefs shared over all operations, and
  // kReadingResourceUse/kWritingResourceUse can be used to represent the
  // shorter lifetime of an in-flight operation.
  kReferencingResourceUse = 1,
  // Reading.
  kReadingResourceUse = 2,
  // Writing.
  kWritingResourceUse = 3,
  // Mutating.
  kMutatingResourceUse = 4
};

class Resource;
using ResourceRef = RCRef<Resource>;

/// An abstract representation of a 'use' of a 'resource'. Manipulated
/// only via the Resource class. Move only with explicit copy.
class ResourceUse {
public:
  /// Constructs the null resource use.
  ResourceUse() = default;

  /// Use explicit copy method for copying.
  ResourceUse(const ResourceUse &) = delete;
  ResourceUse &operator=(const ResourceUse &) = delete;

  /// Move is ok.
  ResourceUse(ResourceUse &&that) { swap(that); }
  ResourceUse &operator=(ResourceUse &&that) {
    swap(that);
    return *this;
  }

  void swap(ResourceUse &rhs);

  StringRef getName() const { return name; }
  ResourceRef getResource() const { return resource.copy(); }
  ResourceUseType getUseType() const { return useType; }
  ResourceSection getSection() const { return section; }

  /// Return copy of this use. The referenced resource is updated
  /// to track the new use.
  ResourceUse copy() const;

  /// Destroys use. The referenced resource is updated to track that this use
  /// has finished.
  ~ResourceUse();

  /// Reset the use to be null.
  void reset();

  void print(llvm::raw_ostream &os) const;
  operator bool() const { return (bool)resource; }

private:
  ResourceUse(std::string name, ResourceRef resource, ResourceUseType useType,
              ResourceSection section)
      : name(std::move(name)), resource(std::move(resource)), useType(useType),
        section(section) {
    assert(useType != kInvalidResourceUse);
    assert(section.end() <= kMaxResourceOffset);
  }

  /// Name of the use, for debugging messages.
  std::string name;
  /// The resource being used, or null.
  ResourceRef resource;
  /// How the resource is used.
  ResourceUseType useType = kInvalidResourceUse;
  /// Section of the resource being used.
  ResourceSection section;

  friend Resource;
};

//===----------------------------------------------------------------------===//
// Resource
//===----------------------------------------------------------------------===//

/// An abstract representation of a 'resource' which may have any number of
/// 'uses' of (sections of) it. It detects:
///  - use after free
///  - data races between concurrent writers and readers/writers.
///  - reading from uninitialized data
///
/// Uses may be of the entire resource or a 'section' of it. Sections cannot
/// overlap.
///
/// This is just all bookkeeping for helping to detect runtime errors, and
/// there's no assumptions about what the resource is or what uses of it
/// actual do. Generally the bookkeeping overhead is way too much for anything
/// other than debug builds.
///
/// Thread safe.
class Resource : public ReferenceCounted<Resource> {
public:
  ~Resource();

  /// No copying or moving.
  Resource(const Resource &) = delete;
  Resource &operator=(const Resource &) = delete;

  /// Creates a resource.
  static RCRef<Resource> allocate(std::string name, bool isInitialized = true);

  StringRef getName() const { return name; }

  /// Returns fresh use of resource. Will assert fail if use conflicts with
  /// other active uses or the current resource state.
  ResourceUse beginUse(std::string useName,
                       ResourceUseType useType = kReferencingResourceUse,
                       ResourceSection section = allResourceSection());

  /// Indicates use has caused resource to become initialized or uninitialized.
  /// Will assert fail if new state conflicts with active use.
  void markInitialized(const ResourceUse &use);
  void markUninitialized(const ResourceUse &use);

  /// Indicates resource is free. Will assert fail if resource still has uses.
  void markFreed();

private:
  Resource(std::string name, bool isInitialized);

  /// Record that given use is about to be destroyed.
  void endUse(const ResourceUse &use);

  /// All the following require the mutex to be held.

  /// Do the freed checks and make the state change.
  void markFreedImpl();

  void print(llvm::raw_ostream &os) const;

  /// Bail out with a hopefully helpful error message.
  void fatal(StringRef message, const ResourceUse &use);
  void fatal(StringRef message);

  /// Maps in-use sections to their bag of use names.
  using UseMap = llvm::DenseMap<ResourceSection, llvm::StringMap<size_t>>;

  /// Record section as being in use within map using useName.
  static void addUseToMap(UseMap &map, const ResourceSection &section,
                          StringRef useName);

  /// Record section as no longer being in use within map using useName.
  /// Returns true if this was the last use of section in map.
  static bool removeUseFromMap(UseMap &map, const ResourceSection &section,
                               StringRef useName);
  enum UsageRule {
    /// Overlap is allowed provided contained by existing.
    kContained,
    /// Overlap is allowed provided exactly equal to existing.
    kEqual,
    /// No overlap with existing is allowed.
    kExclusive,
  };

  /// Checks the map does not already contain a conflicting use of section.
  static ErrorOrSuccess checkForOverlappingSections(
      const UseMap &map, UsageRule usageRule, ResourceUseType existingUseType,
      ResourceUseType desiredUseType, const ResourceSection &section);

  /// Name of resource, for debugging messages.
  std::string name;

  /// Guards all of the following.
  mutable std::mutex mu;

  /// Possible resource states.
  enum ResourceState { kAlive = 0, kFreed = 1 };
  static const char *stateToString(ResourceState state);

  /// Current state of resource.
  ResourceState state = kAlive;
  /// Which sections of resource are 'initialized'.
  ResourceSections initialized;
  /// Active kReferencingResourceUse uses, indexed by the section they are
  /// referencing.
  UseMap referencing;
  /// Active kReadingResourceUse/kMutatingResourceUse uses, indexed by the
  /// section they are reading.
  UseMap reading;
  /// Union of all kReadingResourceUse/kMutatingResourceUse use sections.
  ResourceSections allReading;
  /// Active kWritingResourceUse/kMutatingResourceUse uses, indexed by the
  /// section they are writing. Should only ever contain one key with one use!
  UseMap writing;
  /// Union of all kWritingResourceUse/kMutatingResourceUse use sections.
  ResourceSections allWriting;

  friend ResourceUse;
};

} // namespace M::AsyncRT

namespace std {

// For ADL style swap.
template <>
inline void swap(M::AsyncRT::ResourceUse &lhs, M::AsyncRT::ResourceUse &rhs) {
  lhs.swap(rhs);
}

} // namespace std

#endif // LLCL_SUPPORT_RESOURCE_H
