//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/TypeID.h"
#include "LLCL/Runtime/Globals/TypeInfoTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "typeids"

using namespace M;
using namespace LLCL;

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

static ::TypeInfoTable &getSingleton() {
  return Globals::getTypeInfoTableSingleton(
      [] { return new TypeInfoTable(64); });
}

Detail::RawTypeID TypeID::getSlow(std::string_view typeName,
                                  ValueDestructorFn destructorFn) {
  return getSingleton().getSlow(typeName, destructorFn);
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
  return reinterpret_cast<intptr_t>(&getSingleton());
}

std::string_view TypeID::getTypeName() const {
  return getSingleton().getTypeName(id);
}

ValueDestructorFn TypeID::getValueDestructor() const {
  return getSingleton().getValueDestructor(id);
}
