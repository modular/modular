//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/TypeID.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "LLCL/Support/ConcurrentAppendingVector.h"

#define DEBUG_TYPE "typeids"

using namespace M;
using namespace LLCL;

/// Pair the destructor and type name used at registration time. The latter
/// is very handy for debugging, eg see AsyncValue::printDebug.
struct TypeInfo {
  std::string_view typeName;
  ValueDestructorFn destructorFn;

  TypeInfo(std::string_view typeName, ValueDestructorFn destructorFn)
      : typeName(typeName), destructorFn(destructorFn) {}
};

/// The globally unique type info table. The string -> id mapping uses
/// heavyweight mutex synchronization, but see TypeIDCache for how that cost is
/// amortized. The id -> property mapping only needs atomic synchronization and
/// is very cheap.
struct TypeInfoTable {
  mutable std::mutex m; // protects ids
  llvm::StringMap<Detail::RawTypeID> ids;
  ConcurrentAppendingVector<TypeInfo> entries;

  TypeInfoTable(size_t initialCapacity) : entries(initialCapacity) {}

  Detail::RawTypeID getSlow(std::string_view typeName,
                            ValueDestructorFn destructor);
  std::string_view getTypeName(Detail::RawTypeID id) const {
    return id == Detail::kInvalidRawTypeID ? std::string_view{"unk"}
                                           : entries[id].typeName;
  }
  ValueDestructorFn getValueDestructor(Detail::RawTypeID id) const {
    return id == Detail::kInvalidRawTypeID ? nullptr : entries[id].destructorFn;
  }
};

Detail::RawTypeID TypeInfoTable::getSlow(std::string_view typeName,
                                         ValueDestructorFn destructor) {
  std::lock_guard<std::mutex> l(m);
  auto itr = ids.find(typeName);
  if (itr != ids.end())
    return itr->second;

  size_t id = entries.emplace_back(typeName, destructor);
  assert(id != Detail::kInvalidRawTypeID && "too many type ids registered");
  LLVM_DEBUG(llvm::dbgs() << "Registering type " << typeName << " with " << id
                          << "\n");
  [[maybe_unused]] auto pair = ids.try_emplace(typeName, id);
  assert(pair.second && "already registered type");
  return id;
}

static TypeInfoTable &getTypeInfoTableSingleton() {
  static auto *table = new TypeInfoTable(/*initialCapacity=*/64);
  return *table;
}

Detail::RawTypeID TypeID::getSlow(std::string_view typeName,
                                  ValueDestructorFn destructorFn) {
  return getTypeInfoTableSingleton().getSlow(typeName, destructorFn);
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

intptr_t TypeID::getSignature() {
  return reinterpret_cast<intptr_t>(&getTypeInfoTableSingleton());
}

std::string_view TypeID::getTypeName() const {
  return getTypeInfoTableSingleton().getTypeName(id);
}

ValueDestructorFn TypeID::getValueDestructor() const {
  return getTypeInfoTableSingleton().getValueDestructor(id);
}
