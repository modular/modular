//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_TYPEID_H
#define LLCL_RUNTIME_TYPEID_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

namespace M::LLCL {

/// Type of destructor functions of arbitrary type.
using ValueDestructorFn = void (*)(void *);

namespace Detail {

/// Unfortunately there is no way to build a constexpr string specifically in
/// C++17, and `basic_fixed_string` never made it into C++20.  Just do it the
/// old-fashioned way with a char array rather than implementing a full-fledged
/// `basic_fixed_string` type.
template <std::size_t... Indices>
constexpr auto stringToArray(std::string_view str,
                             std::index_sequence<Indices...>) {
  return std::array{str[Indices]...};
}

template <class T>
constexpr auto typeNameArray() {
#if defined(__clang__)
  constexpr std::string_view prefix = "[T = ";
  constexpr std::string_view suffix = "]";
#elif defined(__GNUC__)
  constexpr std::string_view prefix = "with T = ";
  constexpr std::string_view suffix = "]";
#elif defined(_MSC_VER)
  constexpr std::string_view prefix = "type_name_array<";
  constexpr std::string_view suffix = ">(void)";
#else
#error                                                                         \
    "Modular Runtime built with a toolchain not supporting type introspection."
#endif

  constexpr std::string_view function = LLVM_PRETTY_FUNCTION;

  // The algorithm is straightforward:
  // Find where the prefix starts and record the index at the end of it
  // Find where the suffix ends and record the index at the beginning of it
  // Create a substring between the two indices

  constexpr auto start = function.find(prefix) + prefix.size();
  constexpr auto end = function.rfind(suffix);

  static_assert(start < end,
                "Invalid assumptions about parsing type_name for a type.");

  constexpr auto name = function.substr(start, end - start);
  return stringToArray(name, std::make_index_sequence<name.size()>{});
}

/// In C++17, we can't define an object with static storage duration inside of a
/// `constexpr` function.  However, we can define a `constexpr` object with
/// static storage duration as a member and access through that from within a
/// `constexpr` function.
template <class T>
struct TypeNameHolder {
  static inline constexpr auto value = typeNameArray<T>();
};

/// Currently this only supports getting the demangled type name for a type, and
/// so you cannot specify a non-type (e.g. an NTTP, enum class, etc.) right now.
template <class T>
constexpr std::string_view typeNameFor() {
  constexpr auto &value = TypeNameHolder<T>::value;
  return std::string_view{value.data(), value.size()};
}

/// Returns the 'pretty' name of T for compilers which support it.  Otherwise,
/// gives an error at build time.
///
/// TODO: Should we encounter issues with non-uniqueness of type names (eg
/// because of types in anonymous namespaces) then this template can be
/// specialized, possibly with macro helpers to make it seamless.  We don't
/// currently have this use case, but leaving the door open with this design for
/// now.
///
/// TODO: Should we need to build with toolchains which do not support
/// the PRETTY_FUNCTION machinery then we'll need to register every type
/// manually. See third-party/llvm-project/mlir/include/mlir/Support/TypeID.h.
template <typename T>
struct TypeNameResolver {
  static std::string_view getTypeName() { return typeNameFor<T>(); }
};

/// The ValueDestructorFn for values of type T.
template <typename T>
static void valueDestructorFn(void *pointer) {
  std::destroy_at(static_cast<T *>(pointer));
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
    assert((expected == kInvalidRawTypeID || expected == id) &&
           "inconsistent type ids");
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

  /// Helper function that calls registerType() for each type in the list.
  template <typename... Ts>
  static void registerTypes() {
    (registerType<Ts>(), ...);
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
