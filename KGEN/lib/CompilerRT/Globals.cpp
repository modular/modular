//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT/Registration.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

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

static llvm::SmallDenseMap<llvm::StringRef, GlobalEntry> &getGlobalTable() {
  static llvm::SmallDenseMap<llvm::StringRef, GlobalEntry> table{};
  return table;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_GetGlobalOr(llvm::StringRef name, void *(*initFn)(),
                            void (*destroyFn)(void *)) {
  auto &globalTable = getGlobalTable();

  auto it = globalTable.find(name);
  if (it != globalTable.end())
    return it->second.value;

  GlobalEntry entry(initFn(), destroyFn);
  globalTable.insert({name, entry});

  return entry.value;
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_DestroyGlobals() {
  auto &globalTable = getGlobalTable();
  for (auto entry : globalTable)
    entry.second.destroy();
  globalTable.clear();
}

//===----------------------------------------------------------------------===//
// CompilerRT Registration
//===----------------------------------------------------------------------===//

void M::KGEN::registerGlobals(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back(
      {"KGEN_CompilerRT_GetGlobalOr", (void *)&KGEN_CompilerRT_GetGlobalOr});
  funcs.push_back({"KGEN_CompilerRT_DestroyGlobals",
                   (void *)&KGEN_CompilerRT_DestroyGlobals});
}
