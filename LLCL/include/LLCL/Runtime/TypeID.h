//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_TYPEID_H
#define LLCL_RUNTIME_TYPEID_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"

#include <assert.h>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace M::LLCL {

/// Type of destructor functions of arbitrary type.
using ValueDestructorFn = void (*)(void *);

namespace Detail {

/// Returns the 'pretty' name of T for compilers which support it.
///
/// TODO: Replace with constexpr version.
///
/// TODO: Should we encounter issues with non-uniqueness of type names (eg
/// because of types in anonymous namespaces) then this template can be
/// specialized, possibly with macro helpers to make it seamless.
///
/// TODO: Should we need to build with toolchains which do not support
/// the PRETTY_FUNCTION machinery then we'll need to register every type
/// manually. See third-party/llvm-project/mlir/include/mlir/Support/TypeID.h.
template <typename T>
struct TypeNameResolver {
  static StringRef getTypeName() {
    StringRef nm = llvm::getTypeName<T>();
    assert(
        nm != "UNKNOWN_TYPE" &&
        "The Modular Runtime was built with a toolchain which does not allow "
        "recovery of type names.");
    return nm;
  }
};

/// The ValueDestructorFn for values of type T.
template <typename T>
static void valueDestructorFn(void *pointer) {
  std::destroy_at<T>(static_cast<T *>(pointer));
}

/// The underlying unique 2-byte identifier for a type.
using RawTypeID = uint16_t;

/// The distinguished invalid raw type id.
constexpr RawTypeID kInvalidRawTypeID = RawTypeID(~0);

/// A 'cache' for the raw type id for T. Ok if ends up with
/// compiler-instantiated definitions in multiple dynamic libraries /
/// executable due to template instantiation since the true synchronization is
/// done by the singleton type info table.
///
/// For internal use only.
template <typename T>
struct TypeIDCache {
  static std::atomic<RawTypeID> cachedID;

  /// If an id has not been cached use fn to derive it.
  template <typename Fn>
  static RawTypeID memoize(Fn &&fn) {
    /// Fast path: we've already cached an id in the caller's dynamic library
    /// / executable. Ok to use relaxed consistency since we only care if the
    /// value has already been set.
    RawTypeID id =
        Detail::TypeIDCache<T>::cachedID.load(std::memory_order_relaxed);
    if (id != kInvalidRawTypeID)
      return id;

    /// Slow path: call fn to get the id. We'll use the string name of T to
    /// ensure key uniqueness, and heavyweight synchronization in fn over the
    /// global type info table to ensure id uniqueness.
    id = fn(TypeNameResolver<T>::getTypeName(), &valueDestructorFn<T>);

    /// Cache the id. We don't care if we win the exchange since the
    /// underlying id will be consistent over all threads.
    RawTypeID expected = kInvalidRawTypeID;
    (void)Detail::TypeIDCache<T>::cachedID.compare_exchange_strong(
        expected, id, std::memory_order_relaxed, std::memory_order_relaxed);
    assert(expected == kInvalidRawTypeID ||
           expected == id && "inconsistent type ids");
    return id;
  }
};

template <typename T>
std::atomic<RawTypeID> TypeIDCache<T>::cachedID = kInvalidRawTypeID;

} // namespace Detail

/// A unique (2-byte) identifier for a type.
class TypeID {
public:
  /// Constructs the 'invalid' type id.
  TypeID() = default;

  /// Ensures a unique type id will be available for T. T may have already been
  /// registered. Thread safe. Can be called from multiple dynamic libraries /
  /// executables. Fast after the first call.
  template <typename T>
  static void registerType() {
    (void)Detail::TypeIDCache<T>::memoize(registerTypeSlow);
  }

  /// Returns the unique type id for T. T must have been previously registered.
  /// Thread safe. Can be called from multiple dynamic libraries / executables.
  /// Fast after the first call.
  template <typename T>
  static TypeID get() {
    return TypeID(Detail::TypeIDCache<T>::memoize(getSlow));
  }

  /// Returns a 'signature' for the type id subsystem which is expected to
  /// be unique for the running process. This can be used to catch, at runtime,
  /// accidental multiple definitions for Modular runtime statics across
  /// dynamic libraries / executables.
  ///
  /// (This is just the address of the underlying table info singleton, but
  /// please don't depend on that.)
  static intptr_t getSignature();

  inline bool operator==(const TypeID &other) const { return id == other.id; }
  inline bool operator!=(const TypeID &other) const {
    return !(*this == other);
  }

  /// Returns the name for this type id, or "unk" if invalid.
  StringRef getTypeName() const;

  /// Returns the destructor function for this type id, or null if invalid.
  ValueDestructorFn getValueDestructor() const;

private:
  explicit TypeID(Detail::RawTypeID id) : id(id) {}

  /// Slow path for registerType. Will force global synchronization on global
  /// type info table.
  static Detail::RawTypeID registerTypeSlow(StringRef typeName,
                                            ValueDestructorFn destructorFn);

  /// Slow path for get. Will force global synchronization on global type
  /// info table. The destructorFn argument is ignored, and in present only
  /// for the convenience of TypeIDCache::memoize.
  static Detail::RawTypeID getSlow(StringRef typeName,
                                   ValueDestructorFn destructorFn);

  Detail::RawTypeID id = Detail::kInvalidRawTypeID;
};

} // namespace M::LLCL

#endif // LLCL_RUNTIME_TYPEID_H
