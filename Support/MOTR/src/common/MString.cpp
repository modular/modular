//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "motr/MString.h"
#include "motr/Constants.h"
#include <cassert>

using namespace M::motr;

StringLibrary &MString::getStringLibrary() {
  static StringLibrary stringLibrary;
  static bool once = false;
  if (!once) {
    once = true;
    // add all default strings to the string library
    for (auto sv : Constants::allStringViews)
      stringLibrary.addString(sv);
  }
  return stringLibrary;
}

MString::MString() : hash(0) {
  // allow explicit invalid MString
}

MString::MString(Hash::Value hash, bool failIfNotFound) : hash(hash) {
  // allow explicit invalid MString
  if (hash == 0)
    return;

  assert(!failIfNotFound ||
         valid() && "MString hash is invalid and not found in StringLibrary");
}

MString::MString(std::string_view str) : hash(0) {
  // allow explicit invalid string_view
  if (str.data() == nullptr)
    return;

  auto [sv, hashVal] = getStringLibrary().addString2(str);
  this->hash = hashVal;
  assert(sv == str && "String mismatch");
}

std::string_view MString::sv(bool createPlaceholder) const {
  if (hash.v == 0)
    return {};
  return getStringLibrary().getString(hash.v, createPlaceholder);
}

std::string MString::str(bool createPlaceholder) const {
  // if the string is not found, it will return an empty string
  return std::string{sv(createPlaceholder)};
}

bool MString::valid() const {
  // empty strings are valid, but empty string view data is not
  return sv().data() != nullptr;
}
