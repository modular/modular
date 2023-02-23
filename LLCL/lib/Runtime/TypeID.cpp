//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/TypeID.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#include "LLCL/Support/ConcurrentAppendingVector.h"

#define DEBUG_TYPE "typeids"

using namespace M;
using namespace LLCL;

/// Pair the destructor and type name used at registration time. The latter
/// is very handy for debugging, eg see AsyncValue::printDebug.
struct TypeInfo {
  std::string typeName;
  ValueDestructorFn destructorFn;

  TypeInfo(StringRef typeName, ValueDestructorFn destructorFn)
      : typeName(typeName), destructorFn(destructorFn) {}
};

/// The globally unique type info table. The string -> id mapping uses
/// heavyweight mutex synchronization, but see TypeIDCache for how that cost is
/// amortized. The id -> property mapping only needs atomic synchronization and
/// is very cheep.
struct TypeInfoTable {
  mutable std::mutex m; // protects ids
  llvm::StringMap<Detail::RawTypeID> ids;
  ConcurrentAppendingVector<TypeInfo> entries;

  TypeInfoTable(size_t initialCapacity) : entries(initialCapacity) {}

  Detail::RawTypeID registerTypeSlow(StringRef typeName,
                                     ValueDestructorFn destructor);
  Detail::RawTypeID getSlow(StringRef typeName) const;
  StringRef getTypeName(Detail::RawTypeID id) const {
    return id == Detail::kInvalidRawTypeID ? StringRef("unk", 3)
                                           : entries[id].typeName;
  }
  ValueDestructorFn getValueDestructor(Detail::RawTypeID id) const {
    return id == Detail::kInvalidRawTypeID ? nullptr : entries[id].destructorFn;
  }
};

Detail::RawTypeID
TypeInfoTable::registerTypeSlow(StringRef typeName,
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

Detail::RawTypeID TypeInfoTable::getSlow(StringRef typeName) const {
  std::lock_guard<std::mutex> l(m);
  auto itr = ids.find(typeName);
  if (itr == ids.end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Type " << typeName << " has not been registered\n");
  }
  assert(itr != ids.end() && "type has not been registered");
  return itr->second;
}

static TypeInfoTable &getTypeInfoTableSingleton() {
  static auto *table = new TypeInfoTable(/*initialCapacity=*/64);
  return *table;
}

Detail::RawTypeID TypeID::registerTypeSlow(StringRef typeName,
                                           ValueDestructorFn destructorFn) {
  return getTypeInfoTableSingleton().registerTypeSlow(typeName, destructorFn);
}

Detail::RawTypeID TypeID::getSlow(StringRef typeName,
                                  ValueDestructorFn destructorFn) {
  return getTypeInfoTableSingleton().getSlow(typeName);
}

intptr_t TypeID::getSignature() {
  return reinterpret_cast<intptr_t>(&getTypeInfoTableSingleton());
}

StringRef TypeID::getTypeName() const {
  return getTypeInfoTableSingleton().getTypeName(id);
}

ValueDestructorFn TypeID::getValueDestructor() const {
  return getTypeInfoTableSingleton().getValueDestructor(id);
}
