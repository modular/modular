//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
#define SDK_GRAPHAPI_PYTHON_TYPECASTERS_H

#include "KGEN/KGENDialect/KGENEnums.h"
#include "Support/AssertStream.h"
#include "Support/ML/DType.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/unique_ptr.h"
#include "llvm/Support/raw_ostream.h"

namespace nb = nanobind;

namespace M::Graph::Python {

void registerTypeID(mlir::TypeID, const std::type_info *);
const std::type_info *lookupTypeID(mlir::TypeID);

/// Wrap an OpBuilder->create call.
/// - We can't bind templatized functions on OpBuilder
/// - Instead, each Op has their own constructors that take a builder
/// - Internally, these constructors use `create_op` to delegate to the builder
template <typename Op, typename... Args>
auto create_op(nanobind::handle_t<mlir::OpBuilder> builder, Args... args) {
  return nanobind::cast<mlir::OpBuilder *>(builder)->create<Op>(
      std::forward<Args>(args)...);
}

template <typename T>
class NanobindWrapper {
  // This exists because of typeinfo madness.
  // MLIR doesn't generate typeinfo, and _most_ of its types don't have
  // a key function, so we can generate them ourselves.
  // For a few types, such as mlir::RewriterBase, they _do_ have key functions,
  // and as a result we can't use such types as inputs to our bindings.
  //
  // Instead, we create this wrapper for which we know we can generate typeinfo,
  // and then we codegen usages of it.
public:
  T value;
  NanobindWrapper(T &&value) : value(value) {}
  operator T() { return value; }
};
} // namespace M::Graph::Python

namespace NB_NAMESPACE {
namespace detail {

namespace {
template <std::size_t N>
struct StringLiteral {
  char data[N + 1];
  constexpr StringLiteral(const llvm::StringRef str) {
    for (size_t i = 0; i < N; ++i)
      data[i] = str.data()[i];
    data[N] = '\0';
  }
};

/// Create a nb::const_name from `llvm::getTypeName<T>`
template <typename T>
constexpr auto type_name() {
  constexpr auto name = llvm::getTypeName<T>();
  return nb::detail::const_name(StringLiteral<name.size()>(name).data);
}

//===----------------------------------------------------------------------===//
// is_attribute_interface
//===----------------------------------------------------------------------===//

/// Trait for detecting subclasses of `mlir::AttributeBase
template <typename T>
struct is_attribute_interface_base {
private:
  template <typename C, typename Concrete, typename... Traits>
  static std::true_type
  test(const mlir::AttributeInterface<Concrete, Traits...> *);

  template <typename C>
  static std::false_type test(...);

public:
  using type = decltype(test<T>(std::declval<T *>()));
  static constexpr bool value = type::value;
};

/// Trait function for detecting subclasses of `mlir::AttributeBase
template <typename T>
constexpr bool is_attribute_interface() {
  return is_attribute_interface_base<T>::value;
}
} // namespace

//===----------------------------------------------------------------------===//
// TypeCasters for LLVM and MLIR types
//===----------------------------------------------------------------------===//

/// Casts object <-> MLIRContext.
/// Only passes by pointer; we never want to take or store an MLIRContext by
/// value, which will deep copy all of the storage data.
template <>
struct type_caster<::mlir::MLIRContext> {
protected:
  ::mlir::MLIRContext *value;

public:
  static constexpr auto Name = const_name("Context");

  template <typename T>
  using Cast = ::mlir::MLIRContext *;

  operator ::mlir::MLIRContext *() { return value; }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    nb::object capsule;
    try {
      capsule = mlirApiObjectToCapsule(src);
    } catch (nb::builtin_exception) {
      return false;
    }
    value = unwrap(mlirPythonCapsuleToContext(capsule.ptr()));
    return !mlirContextIsNull(wrap(value));
  }
};

/// Casts MlirLocation <-> mlir::Location.
/// Delegate to the `MlirLocation` upstream type caster.
template <>
struct type_caster<::mlir::Location> {
  static constexpr auto Name = const_name("Location");
  template <typename T>
  using Cast = movable_cast_t<::mlir::Location>;
  using Caster = make_caster<MlirLocation>;
  Caster caster;

  /// Since `mlir::Location` doesn't have a default constructor, store the value
  /// as an `optional` until it is successfully cast.
  std::optional<::mlir::Location> value;

  explicit operator ::mlir::Location *() { return &*value; }
  explicit operator ::mlir::Location &() { return *value; }
  explicit operator ::mlir::Location &&() { return std::move(*value); }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = unwrap(caster.value);
    return true;
  }

  template <typename T>
  static constexpr bool can_cast() {
    return Caster::can_cast<T>();
  }

  static handle from_cpp(::mlir::Location t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(wrap(t), policy, cleanup);
  }
};

/// Casts str <-> llvm::StringRef.
/// Delegate implementation to the `std::string_view` caster.
template <>
struct type_caster<::llvm::StringRef> {
  NB_TYPE_CASTER(::llvm::StringRef, const_name("str"))
  using Caster = make_caster<std::string_view>;
  Caster caster;

  bool from_python(handle_t<nb::str> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = caster.value;
    return true;
  }

  static handle from_cpp(::llvm::StringRef t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(t, policy, cleanup);
  }
};

/// Casts int <-> llvm::APInt.
/// There's likely a _much_ better way to do this by directly passing
/// the int bytes between bignum implementations. For now stringifying and
/// parsing.
template <>
struct type_caster<::llvm::APInt> {
  NB_TYPE_CASTER(::llvm::APInt, const_name("int"))

  bool from_python(handle_t<nb::int_> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    auto base10 = nb::cast<std::string>(nb::str(src));
    llvm::StringRef(base10).getAsInteger(10, value);
    return true;
  }

  static handle from_cpp(::llvm::APInt t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    std::string base10;
    llvm::raw_string_ostream(base10) << t;
    // _Very_ important to release here.
    // - In most of our type casters we defer to another caster, or explicitly
    // pass ownership of a C++ type via a `unique_ptr`
    // - Here we are returning an `nb::object` directly. `release` gives
    // ownership to python.
    // - Otherwise, you can end up double-freeing memory from Python's interned
    // longs, which is a spooky and very hard to track down memory safety bug.
    return nb::int_(make_caster<std::string>::from_cpp(base10, policy, cleanup))
        .release();
  }
};

/// Casts str <-> llvm::Twine.
/// Twines are meant to be temporary values, so we treat them
/// more like values than stringrefs. We don't support taking pointers to them.
/// - Strings passed from Python -> C++ as a Twine will not copy since we can
/// expose a single stringref as a Twine
/// - Twines passed from C++ -> Python copy into a contiguous string and are
/// passed to Python.
template <>
struct type_caster<::llvm::Twine> {
protected:
  std::string_view value;

public:
  static constexpr auto Name = const_name("str");

  template <typename T>
  using Cast = ::llvm::Twine;

  operator ::llvm::Twine() { return value; }
  using Caster = make_caster<std::string_view>;
  Caster caster;

  bool from_python(handle_t<nb::str> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = caster.value;
    return true;
  }

  static handle from_cpp(::llvm::Twine t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return make_caster<std::string>::from_cpp(t.str(), policy, cleanup);
  }
};

/// Casts object <-> DType.
template <>
struct type_caster<::M::DType> {
  NB_TYPE_CASTER(::M::DType, const_name("max._core.dtype.DType"))
  using DTypeCaster = make_caster<M::DType::Cases>;
  DTypeCaster dtypeCaster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (dtypeCaster.from_python(src, flags, cleanup)) {
      value = M::DType(dtypeCaster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::M::DType t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return DTypeCaster::from_cpp(static_cast<M::DType::Cases>(t.getValue()),
                                 policy, cleanup);
  }
};

/// Casts AttributeInterface <-> python.
/// - General type caster for any attribute interfaces
/// - Allows any type implementing the interface to be passed Python -> C++
/// - Downcasts to the concrete attribute implementation type when passed C++ ->
/// Python
template <typename AttributeInterface>
struct type_caster<
    AttributeInterface,
    std::enable_if_t<is_attribute_interface<AttributeInterface>(), int>> {
  NB_TYPE_CASTER(AttributeInterface, const_name("\"") +
                                         type_name<AttributeInterface>() +
                                         const_name("\""))

  using Caster = make_caster<mlir::Attribute>;
  Caster caster;

  bool from_python(handle_t<::mlir::Attribute> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup))
      return false;
    value =
        ::mlir::dyn_cast_or_null<AttributeInterface>(mlir::Attribute(caster));
    return bool(value);
  }

  static handle from_cpp(AttributeInterface ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return make_caster<std::unique_ptr<::mlir::Attribute>>::from_cpp(
        std::make_unique<::mlir::Attribute>(ar), policy, cleanup);
  }
};

/// Downcast known Attributes to their bound type object.
/// If we don't know it, return the base Attribute.
template <>
struct type_hook<::mlir::Attribute> {
  static const std::type_info *get(::mlir::Attribute *attr) {
    if (attr && *attr) {
      if (auto info = M::Graph::Python::lookupTypeID(attr->getTypeID()))
        return info;
    }
    return &typeid(::mlir::Attribute);
  }
};

/// Downcast known Types to their bound type object.
/// If we don't know it, return the base Type.
template <>
struct type_hook<::mlir::Type> {
  static const std::type_info *get(::mlir::Type *type) {
    if (type && *type) {
      if (auto info = M::Graph::Python::lookupTypeID(type->getTypeID()))
        return info;
    }
    return &typeid(::mlir::Type);
  }
};

/// Casts sequence <-> ArrayRef.
/// This currently copies in each direction.
/// - For Python -> C++ it's unlikely we could improve this, except in the case
/// where the Python value already represents a contiguous C++ array.
/// - For C++ -> Python, we can eventually return a special type which wraps the
/// ArrayRef as a Sequence type, and could be passed back to C++ as an ArrayRef.
/// Care needs to be taken with the lifetime of this reference.
template <typename Entry>
struct type_caster<::llvm::ArrayRef<Entry>> {
  using Caster = make_caster<Entry>;
  using VecCaster = make_caster<std::vector<Entry>>;
  NB_TYPE_CASTER(::llvm::ArrayRef<Entry>,
                 const_name("collections.abc.Sequence[") + Caster::Name +
                     const_name("]"))

  VecCaster caster;
  bool used = false;

  bool from_python(handle_t<nb::sequence> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    // Nanobind's built in typecasters for collections re-use
    // their internal typecasters, which isn't safe for array refs.
    // We can do something smart like specialize the caster for
    // std::vector<ArrayRef<T>> to hold a vector of typecaster instances.
    ASSERT_STREAM(!used, "ArrayRef typecasters cannot be reused.");
    if (!caster.from_python(src, flags, cleanup))
      return false;
    used = true;
    const Entry *start = caster.value.data();
    value = ::llvm::ArrayRef<Entry>(start, caster.value.size());
    return true;
  }
  static handle from_cpp(::llvm::ArrayRef<Entry> ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    // TODO(MAXPLAT-123): Make these views.
    return VecCaster::from_cpp(ar.vec(), policy, cleanup);
  }
};

/// Casts sequence[bool] <-> ArrayRef<bool>.
/// The default implementation doesn't work because std::vector<bool>
/// is specialized as a bit vector, which we can't take a reference to.
template <>
struct type_caster<::llvm::ArrayRef<bool>> {
  using Caster = make_caster<bool>;
  NB_TYPE_CASTER(::llvm::ArrayRef<bool>,
                 const_name("collections.abc.Sequence[bool]"))

  Caster caster;
  ::llvm::SmallVector<bool> storage;
  bool used = false;

  bool from_python(handle_t<nb::sequence> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    // Nanobind's built in typecasters for collections re-use
    // their internal typecasters, which isn't safe for array refs.
    // We can do something smart like specialize the caster for
    // std::vector<ArrayRef<T>> to hold a vector of typecaster instances.
    ASSERT_STREAM(!used, "ArrayRef typecasters cannot be reused.");
    for (auto entry : src) {
      if (!caster.from_python(entry, flags, cleanup)) {
        storage.clear();
        return false;
      }
      storage.push_back(caster.value);
    }
    used = true;
    const bool *start = storage.data();
    value = ::llvm::ArrayRef<bool>(start, storage.size());
    return true;
  }

  static handle from_cpp(::llvm::ArrayRef<bool> ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    // TODO(MAXPLAT-123): Make these views.
    return make_caster<std::vector<bool>>::from_cpp(ar.vec(), policy, cleanup);
  }
};

/// Casts object <-> mlir::TypeRange.
/// This makes a copy of the type pointers passing either direction.
template <>
struct type_caster<::mlir::TypeRange> {
  using Caster = make_caster<llvm::ArrayRef<mlir::Type>>;
  NB_TYPE_CASTER(::mlir::TypeRange, Caster::Name)
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = mlir::TypeRange(caster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::mlir::TypeRange t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::VecCaster::from_cpp(t, policy, cleanup);
  }
};

/// Casts object <-> mlir::ValueRange.
/// This makes a copy of the value pointers passing either direction.
template <>
struct type_caster<::mlir::ValueRange> {
  using Caster = make_caster<llvm::ArrayRef<mlir::Value>>;
  NB_TYPE_CASTER(::mlir::ValueRange, Caster::Name)
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = mlir::ValueRange(caster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::mlir::ValueRange t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::VecCaster::from_cpp(t, policy, cleanup);
  }
};

/// Casts object <-> M::KGEN::FnEffects.
/// - C++ has M::KGEN::impl::FnEffects and M::KGEN::FnEffects
/// - We just use one for simplicity, and standardize on the generated one.
template <>
struct type_caster<M::KGEN::FnEffects> {
  using Caster = make_caster<M::KGEN::impl::FnEffects>;
  NB_TYPE_CASTER(M::KGEN::FnEffects, Caster::Name)
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = caster.value;
      return true;
    }
    return false;
  }
  static handle from_cpp(M::KGEN::FnEffects t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(t.getImpl(), policy, cleanup);
  }
};

} // namespace detail
} // namespace NB_NAMESPACE

#endif // SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
