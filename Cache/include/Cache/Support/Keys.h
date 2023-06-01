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
#include "Support/Host.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
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

struct StringHashedKey {
public:
  using KeyTy = StringRef;
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

struct HostInfoWrapper {
  static std::string wrapKey(const std::string &keyToBeWrapped) {
    std::string hostInfo;
    llvm::raw_string_ostream os(hostInfo);
    auto machineInfoOr = M::getHostMachineInfo();
    if (machineInfoOr.isError())
      return "";
    HostMachineInfo machineInfo = machineInfoOr.takeValue();
    machineInfo.print(os);
    return keyToBeWrapped + hostInfo;
  }
};

struct HostStaticInfoWrapper {
  static std::string wrapKey(const std::string &keyToBeWrapped) {
    std::string features;
    auto machineInfoOr = M::getHostMachineInfo();
    if (machineInfoOr.isError())
      return "";
    HostMachineInfo machineInfo = machineInfoOr.takeValue();
    std::string hostInfo;
    llvm::raw_string_ostream os(hostInfo);
    machineInfo.printStaticInfo(os);
    return keyToBeWrapped + hostInfo;
  }
};

struct CPUFeatureWrapper {
  /// Matrix of supported targets is ARCHS * FEATURES
  /// User is responsible for not tagging with incompatible combinations
  /// (`apple-m1` is omitted as an arch, we cannot access it's cpu features to
  /// cache artifacts)
  static constexpr std::array<StringLiteral, 2> SUPPORTED_ARCHS = {"x86_64",
                                                                   "arm64"};

  /// These advanced CPU features are sorted in decreasing order of "strength".
  /// If multiple CPU features exist we choose the strongest one. These features
  /// are for retrieving hardware-specific libs and CXXDriverAPI has in place a
  /// mechanism for gracefully downgrading to less performant libs in the
  /// absence of these features (See CXXDriverAPI.cpp : getDriverWithPreference)
  static constexpr std::array<StringLiteral, 3> SUPPORTED_FEATURES = {
      "avx512f", "avx2", "avx"};

  static std::string wrapKey(const std::string &keyToBeWrapped) {
    auto machineInfoOr = M::getHostMachineInfo();
    if (machineInfoOr.isError())
      return "";
    auto machineInfo = *machineInfoOr;
    auto features = machineInfo.cpuFeatures;

    auto cpuArchWDelimiter =
        llvm::Triple(machineInfo.triple).getArchName().str() + ":";

    /// We iterate on features in decreasing order of strength. We wrap with the
    /// strongest feature found on the target machine.
    for (auto supportedFeature : SUPPORTED_FEATURES) {
      if (llvm::is_contained(features, supportedFeature))
        return (keyToBeWrapped + cpuArchWDelimiter + supportedFeature).str();
    }

    return keyToBeWrapped;
  }
};

/// Wrap a given key generator with one or more wrappers. Wrappers need to
/// implement a static function wrapKey which takes a string and returns a
/// string back. Wrapping works like this.
///        1. Generate key with key generator
///        2. Call Wrappers::wrapKey(...) on generated key.
///        3. Repeat for all wrappers with previous result.
///        4. Once again hash the final accumulated string.
/// Wrapping takes place in the order in which it is defined
template <typename KeyGen, typename... Wrappers>
class WrappedKey {
public:
  using KeyTy = typename KeyGen::KeyTy;

  static std::string hashKey(KeyTy key) {
    std::string hashedKey = KeyGen::hashKey(std::forward<KeyTy>(key));
    // Apply all the wrappers.
    ([&]() mutable {
      hashedKey = Wrappers::wrapKey(hashedKey);
      return true;
    }() &&
     ...);
    llvm::BLAKE3 hashState{};
    hashState.update(hashedKey);
    auto hash = hashState.final();
    return {hash.begin(), hash.end()};
  }
};

/// A Key Generator that qualifies a given key with host machine info. This is
/// useful in cases where there is one common parent cache object and this
/// produces target specific cache artifacts. This key can be used to lookup
/// and add objects to cache related to the parent but is specific to current
/// host machine. Note that this key doesn't have semantic understanding. So
/// if 2 machines have same cpu features but are ordered differently the hash
/// key will be different. That being said, the machinery used underneath to
/// generate host info sorts the features alphabetically to try to maintain
/// consistency.
template <typename TyKey>
using KeyWithHostInfo = WrappedKey<TyKey, HostInfoWrapper>;

/// This key is similar to KeyWithHostInfo but does not contain information
/// about number of cores or thread affinities.
template <typename TyKey>
using KeyWithStaticHostInfo = WrappedKey<TyKey, HostStaticInfoWrapper>;

/// This is a key that is wrapped based on CPU features detected at runtime.
/// This is useful for retrieving artifacts that are optimized for that CPU
/// feature. Currently supports detection of features AVX2 and AVX512 only.
template <typename TyKey>
using CPUFeatureWrappedKey = WrappedKey<TyKey, CPUFeatureWrapper>;

/// Provide a key that doesn't do any hashing - we only want to read things
/// from keys provided to this.
using ReadOnlyKey = TypeKey<llvm::StringRef>;
} // namespace M::Cache::Keys

#endif // CACHE_SUPPORT_KEYS_H
