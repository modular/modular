//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of commonly used keys and generic type
// infra to create custom keys by composition.
//
//===----------------------------------------------------------------------===//

#ifndef CACHE_SUPPORT_KEYS_H
#define CACHE_SUPPORT_KEYS_H

#include "Cache/Buffer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <string>
#include <variant>

namespace M::Cache::Keys {
template <typename T>
struct TypeKey : std::false_type {};

/// A simple key that takes a StringRef and returns the string from it
/// without any hashing.
template <>
struct TypeKey<llvm::StringRef> {
  using KeyTy = llvm::StringRef;
  static std::string hashKey(KeyTy key) { return key.str(); }
};

template <>
struct TypeKey<llvm::ArrayRef<uint8_t>> {
  using KeyTy = llvm::ArrayRef<uint8_t>;
  static std::string hashKey(KeyTy key) {
    llvm::BLAKE3 hashState{};
    hashState.update(key);

    auto hash = hashState.final();
    return {hash.begin(), hash.end()};
  }
};

template <>
struct TypeKey<M::Cache::BufferRef> {
  using KeyTy = M::Cache::BufferRef;
  static std::string hashKey(KeyTy key) {
    llvm::BLAKE3 hashState{};
    hashState.update(key->getBuffer());

    auto hash = hashState.final();
    return {hash.begin(), hash.end()};
  }
};

template <typename... Ts>
struct VariantTypeKey {
  using KeyTy = std::variant<Ts...>;

  static std::string hashKey(KeyTy key) {
    std::string hashedKey;

    // Go through the types and if any of them belongs to the variant
    // get key for it.
    (getUnderlyingHash<Ts>(std::forward<KeyTy>(key), hashedKey) || ...);
    return hashedKey;
  }

private:
  template <typename T>
  static bool getUnderlyingHash(KeyTy key, std::string &out) {
    if (std::holds_alternative<T>(std::forward<KeyTy>(key))) {
      out = TypeKey<T>::hashKey(std::get<T>(std::forward<KeyTy>(key)));
      return true;
    }
    return false;
  }
};

/// Provide a key that doesn't do any hashing - we only want to read things from
/// keys provided to this.
using ReadOnlyKey = TypeKey<llvm::StringRef>;
} // namespace M::Cache::Keys

#endif // CACHE_SUPPORT_KEYS_H
