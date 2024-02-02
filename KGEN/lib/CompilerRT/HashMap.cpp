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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <mutex>

#define DEBUG_TYPE "mojo-hashmap"

// TODO: These are added for ModCon demo of tokenizer and needs to go away once
// once we have generic hashmaps in Mojo. The reason these are added to
// to CompilerRT so the functions are available at graph execution time also
// without loading a dylib.

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void *
KGEN_CompilerRT_NewStringIntMap() {
  return new llvm::StringMap<ssize_t>();
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_DeleteStringIntMap(void *ptr) {
  delete static_cast<llvm::StringMap<ssize_t> *>(ptr);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT void
KGEN_CompilerRT_InsertIntoStringIntMap(void *ptr, llvm::StringRef key,
                                       ssize_t value) {
  auto map = static_cast<llvm::StringMap<ssize_t> *>(ptr);
  map->insert({key, value});
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT ssize_t
KGEN_CompilerRT_GetFromStringIntMap(void *ptr, llvm::StringRef key) {
  auto map = static_cast<llvm::StringMap<ssize_t> *>(ptr);
  return map->at(key);
}

COMPILERRT_EXPORT COMPILERRT_VISIBILITY_EXPORT int
KGEN_CompilerRT_ExistInStringIntMap(void *ptr, llvm::StringRef key) {
  auto map = static_cast<llvm::StringMap<ssize_t> *>(ptr);
  return map->contains(key);
}

//===----------------------------------------------------------------------===//
// CompilerRT Registration
//===----------------------------------------------------------------------===//

void M::KGEN::registerHashMap(
    std::vector<std::pair<llvm::StringLiteral, void *>> &funcs) {
  funcs.push_back({"KGEN_CompilerRT_NewStringIntMap",
                   (void *)&KGEN_CompilerRT_NewStringIntMap});
  funcs.push_back({"KGEN_CompilerRT_DeleteStringIntMap",
                   (void *)&KGEN_CompilerRT_DeleteStringIntMap});
  funcs.push_back({"KGEN_CompilerRT_InsertIntoStringIntMap",
                   (void *)&KGEN_CompilerRT_InsertIntoStringIntMap});
  funcs.push_back({"KGEN_CompilerRT_GetFromStringIntMap",
                   (void *)&KGEN_CompilerRT_GetFromStringIntMap});
  funcs.push_back({"KGEN_CompilerRT_ExistInStringIntMap",
                   (void *)&KGEN_CompilerRT_ExistInStringIntMap});
}
