//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ADT_GLOBALTABLE_H
#define SUPPORT_ADT_GLOBALTABLE_H

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"

#include "llvm/Support/ErrorHandling.h"
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>

template <>
struct llvm::DenseMapInfo<std::string> {
  static inline std::string getEmptyKey() { return std::string(""); }
  static inline std::string getTombstoneKey() { return std::string(); }
  static unsigned getHashValue(const std::string &ref) {
    return llvm::hash_value(ref);
  }
  static bool isEqual(const std::string &lhs, const std::string &rhs) {
    return lhs == rhs;
  }
};

namespace M {

/// OverflowGlobalEntry represents an entry for the GlobalTable's
/// overflow MapVector when the hash table is full.
struct OverflowGlobalEntry {
  void *value;
  void (*destroyFn)(void *);

  OverflowGlobalEntry(void *value, void (*destroyFn)(void *))
      : value(value), destroyFn(destroyFn) {}

  void destroy() {
    if (destroyFn && value)
      destroyFn(value);
  }
}; // struct OverflowGlobalEntry

/// GlobalTable is a generic typeless storage used by Mojo's CompilerRT
/// interface to sys.ffi._Globals.
//
/// This implementation uses a hybrid approach with fixed size lock-free hash
/// map for the main table and a mutex-protected overflow container when the
/// hash table exceeds capacity.
struct GlobalTable {
  void *getOrCreate(llvm::StringRef name, void *(*initFn)(),
                    void (*destroyFn)(void *));

  void

  clear();

private:
  struct LockFreeGlobalEntry;
  void insertIntoOrderList(LockFreeGlobalEntry *entry);
  void *getFromOverflow(llvm::StringRef name) const;
  void *getOrCreateInOverflow(llvm::StringRef name, void *(*initFn)(),
                              void (*destroyFn)(void *));

  static constexpr size_t kTableSize = 4096;
  static constexpr size_t kMaxProbes = 12;

  std::array<std::atomic<LockFreeGlobalEntry *>, kTableSize> hashTable{};
  std::atomic<LockFreeGlobalEntry *> orderHead{nullptr};

  // The overflowTable container is used when the hashTable capacity is reached.
  std::atomic<bool> hasOverflowEntries{false};
  mutable std::mutex overflowMutex;
  llvm::MapVector<std::string, OverflowGlobalEntry> overflowTable;
}; // struct GlobalTable

} // namespace M

#endif // SUPPORT_ADT_GLOBALTABLE_H
