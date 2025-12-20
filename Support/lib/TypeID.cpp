//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/TypeID.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <mutex>
#include <string_view>

#define DEBUG_TYPE "typeids"

using namespace M;

Detail::RawTypeID Detail::TypeInfoTable::getSlow(std::string_view typeName,
                                                 ValueDestructorFn destructor) {
  std::lock_guard<std::mutex> l(mu);
  auto itr = ids.find(typeName);
  if (itr != ids.end())
    return itr->second;

  size_t id = entries.emplace_back(typeName, destructor);
  assert(id != Detail::kInvalidRawTypeID && "too many type ids registered");
  LDBG() << "Registering type " << typeName << " with " << id;
  [[maybe_unused]] auto pair = ids.try_emplace(typeName, id);
  assert(pair.second && "already registered type");
  return id;
}

Detail::RawTypeID TypeID::getSlow(std::string_view typeName,
                                  ValueDestructorFn destructorFn) {
  return Detail::TypeInfoTable::getSingleton().getSlow(typeName, destructorFn);
}

#if MODULAR_DEBUG
void TypeID::printErrorIfNotEqual(TypeID expected, StringRef context) const {
  if (id == expected.id)
    return;
  llvm::errs() << context << ": object has actual runtime type '"
               << getTypeName()
               << "' however it was expected at compile time to have type '"
               << expected.getTypeName() << "'\n";
}
#endif
