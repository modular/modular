//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "Support/SymbolExport.h"

#include <atomic>
#include <mutex>

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

namespace {
struct GlobalEntry {
  void *value;
  void (*destroyFn)(void *);

  GlobalEntry(void *value, void (*destroyFn)(void *))
      : value(value), destroyFn(destroyFn) {}

  void destroy() {
    if (destroyFn && value)
      destroyFn(value);
  }
};
} // namespace

using GlobalTable = llvm::MapVector<std::string, GlobalEntry>;

/// Note that we want this to be ordered because when destructuring we want to
/// to destroy the first element that was inserted last.
static GlobalTable &getGlobalTable() {
  static GlobalTable globalTable;
  return globalTable;
}

static std::mutex &getGlobalTableMutex() {
  static std::mutex mu; // Serialize global table mutation.
  return mu;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetOrCreateGlobal(llvm::StringRef name, void *(*initFn)(),
                                  void (*destroyFn)(void *)) {
  auto &globalTable = getGlobalTable();

  {
    std::lock_guard<std::mutex> l(getGlobalTableMutex());
    auto it = globalTable.find(name.str());
    if (it != globalTable.end())
      return it->second.value;
  }

  if (!initFn)
    return nullptr;

  GlobalEntry entry(initFn(), destroyFn);

  GlobalTable::iterator itr;
  bool inserted;
  {
    std::lock_guard<std::mutex> l(getGlobalTableMutex());
    std::tie(itr, inserted) = globalTable.insert({name.str(), entry});
  }

  if (!inserted) {
    entry.destroy();
    return itr->second.value;
  }

  return entry.value;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetGlobalOrNull(llvm::StringRef name) {
  return KGEN_CompilerRT_GetOrCreateGlobal(name, nullptr, nullptr);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_InsertGlobal(llvm::StringRef name, void *value) {
  auto &globalTable = getGlobalTable();

  std::lock_guard<std::mutex> l(getGlobalTableMutex());
  globalTable.insert({name.str(), GlobalEntry(value, nullptr)});
}

//===----------------------------------------------------------------------===//
// Indexed globals for well known constants.
//===----------------------------------------------------------------------===//

namespace {
struct AtomicGlobalEntry {
  std::atomic<void *> value;
  std::atomic<void (*)(void *)> destroyFn;

  void destroy() {
    if (auto loadedValue = value.load()) {
      value.store(nullptr);
      if (auto loadedDestroyFn = destroyFn.load()) {
        destroyFn.store(nullptr);
        loadedDestroyFn(loadedValue);
      }
    }
  }
};
} // namespace

/// Keep this as big as the indexed globals in ffi.mojo.
#define NUM_INDEXED_GLOBALS 2
static AtomicGlobalEntry indexedTable[NUM_INDEXED_GLOBALS];

/// A faster version of GetOrCreateGlobal that doesn't need to lock the table
/// or hash the name.
COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetOrCreateGlobalIndexed(size_t index, void *(*initFn)(),
                                         void (*destroyFn)(void *)) {
  assert(index < NUM_INDEXED_GLOBALS && "Unsupported indexed global #");

  // Most accesses will be initialized.
  auto entry = indexedTable[index].value.load();
  if (entry)
    return entry;

  // If not, create a value.
  auto newValue = initFn();
  // Try to swap it in, replacing a nullptr.
  if (!indexedTable[index].value.compare_exchange_strong(entry, newValue)) {
    // If we raced and someone else won, delete whatever we just created.
    destroyFn(newValue);
    return entry;
  }
  // Unconditionally set the destroy function. It should always be the same for
  // anyone racing on this.
  indexedTable[index].destroyFn.store(destroyFn);
  return newValue;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_DestroyGlobals() {
  auto &globalTable = getGlobalTable();
  // Loop in reverse. The reason is say you load a library (using dlopen) and
  // then want to call a function in the library to destroy another global. Then
  // you want to make sure that dlclose happens last.
  for (auto entry : llvm::reverse(globalTable))
    entry.second.destroy();
  globalTable.clear();

  // Destroy indexed globals last.
  for (auto &entry : indexedTable)
    entry.destroy();
}
