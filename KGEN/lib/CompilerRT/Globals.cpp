//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

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
KGEN_CompilerRT_GetGlobalOrCreate(llvm::StringRef name, void *payload,
                                  void *(*initFn)(void *),
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

  GlobalEntry entry(initFn(payload), destroyFn);

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
  return KGEN_CompilerRT_GetGlobalOrCreate(name, nullptr, nullptr, nullptr);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_InsertGlobal(llvm::StringRef name, void *value) {
  auto &globalTable = getGlobalTable();

  std::lock_guard<std::mutex> l(getGlobalTableMutex());
  globalTable.insert({name.str(), GlobalEntry(value, nullptr)});
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
}

//===----------------------------------------------------------------------===//
// CompilerRT Registration
//===----------------------------------------------------------------------===//

void M::KGEN::registerGlobals(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_GetGlobalOrCreate",
                   (void *)&KGEN_CompilerRT_GetGlobalOrCreate});
  funcs.push_back({"KGEN_CompilerRT_GetGlobalOrNull",
                   (void *)&KGEN_CompilerRT_GetGlobalOrNull});
  funcs.push_back(
      {"KGEN_CompilerRT_InsertGlobal", (void *)&KGEN_CompilerRT_InsertGlobal});
  funcs.push_back({"KGEN_CompilerRT_DestroyGlobals",
                   (void *)&KGEN_CompilerRT_DestroyGlobals});
}
