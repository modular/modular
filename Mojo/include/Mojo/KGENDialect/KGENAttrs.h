//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file defines the core KGEN attribute classes, provides implementation
// logic for working with them, and helpers for defining operations that take
// them.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENATTRS_H
#define KGEN_KGENDIALECT_KGENATTRS_H

#include "Mojo/KGENDialect/KGENAttrInterfaces.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"

namespace mlir {
class OperationName;
} // namespace mlir

namespace M {
class TargetInfoAttr;

namespace KGEN {
class BuildInfoType;
class ConformanceOp;
class FuncOp;
class GeneratorOp;
class ParameterEvaluator;
class TargetType;
class ParamListType;
class ParamListAttr;
class SIMDType;
class StructDeclInterface;
} // namespace KGEN
} // namespace M

//===----------------------------------------------------------------------===//
// DTypeValue
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// This class stores a value of a particular dtype. It supports containing
/// integer, index, float, and bool dtype values only. Index values are treated
/// as signed.
class DTypeValue {
public:
  /// Get an integer value. Signedness is taken from `dtype`, not `value`.
  DTypeValue(APSInt value, KGENDType dtype);

  /// Get a floating point value.
  DTypeValue(APFloat value, KGENDType dtype);

  /// Get a bool value.
  DTypeValue(bool value, KGENDType dtype);

  /// Get an index value.
  DTypeValue(int64_t value, KGENDType dtype);

  /// Raw data constructor.
  DTypeValue(APInt data, KGENDType dtype);

  /// Bit width is ignored: it differs across targets for index-like dtypes.
  bool operator==(const DTypeValue &rhs) const {
    return dtype == rhs.dtype && APSInt::isSameValue(data, rhs.data);
  }

  /// Get the underlying data.
  const APSInt &getData() const { return data; }

  /// Get the dtype.
  KGENDType getDType() const { return dtype; }

  /// Get the value as an integer.
  APSInt getIntVal() const;

  /// Get the value as a float.
  APFloat getFloatVal() const;

  /// Get the value as a bool
  bool getBoolVal() const;

  /// Get the value as an index.
  int64_t getIndexVal() const;

private:
  /// Default constructor accessible only by the attribute storage class.
  DTypeValue() {}

  /// All values are stored as `APSInt`s, signed iff the dtype is.
  APSInt data;

  /// The dtype of the value. This indicates how to interpret `data`.
  KGENDType dtype;
};

namespace detail {
struct SIMDAttrStorage;
} // namespace detail

/// Format a single DTypeValue element to os according to dtype
void printDTypeValue(llvm::raw_ostream &os, const DTypeValue &value,
                     KGENDType dtype);

/// Print an array of DTypeValues to os: a bare scalar for a single-element
/// array, or a bracketed comma-separated list for wider ones.
void printDTypeValues(llvm::raw_ostream &os, llvm::ArrayRef<DTypeValue> values,
                      KGENDType dtype);

} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Mojo/KGENDialect/KGENAttrs.h.inc"

//===----------------------------------------------------------------------===//
// EmitAsAttr
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class EmitAsAttr : public IntegerAttr {
public:
  using IntegerAttr::IntegerAttr;
  static bool classof(Attribute attr);
  static EmitAsAttr get(MLIRContext *ctx, EmitAs val);
  EmitAs getValue() const;
};

//===----------------------------------------------------------------------===//
// Sugar Processing for Type and Attribute
//===----------------------------------------------------------------------===//
//
// SugarAttr represents a "syntax sugar" on a type or typed attr, e.g. when
// resolving "alias four = 4", we might want to preserve the name "four" instead
// of inlining the value.  These are typically looked through by semantic
// analysis, but used when generating user-visible error messages.
//
// SugarAttr is lowered away by LowerLIT.

/// Given an attribute or type, return the "canonical" version of the attribute
/// with all type sugar removed.
Attribute getCanonicalAttr(Attribute src);
TypedAttr getCanonicalAttr(TypedAttr src);
Type getCanonicalType(Type type);

/// Return true if the specified types are canonically equal.
bool isEqualCanon(Type t1, Type t2);
bool isEqualCanon(TypedAttr ta1, TypedAttr ta2);

template <typename T>
constexpr bool isValidSugarCastType =
    (std::is_convertible_v<T, Attribute> || std::is_convertible_v<T, Type>);

// Helpers for sugar-aware casting.
template <typename... To, typename From>
[[nodiscard]] inline bool sugarIsa(From val) {
  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  auto stripped = SugarAttr::strip(val);
  return (isa<To>(stripped) || ...);
}

// Helpers for sugar-aware casting.
template <typename... To, typename From>
[[nodiscard]] inline bool sugarIsaAndNonNull(From val) {
  if (!val)
    return false;

  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  val = SugarAttr::strip(val);
  return (isa<To>(val) || ...);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarCast(From val) {
  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  return cast<To>(SugarAttr::strip(val));
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarDynCast(From val) {
  static_assert(isValidSugarCastType<From>,
                "sugared casts only work with Type and Attribute");
  return dyn_cast<To>(SugarAttr::strip(val));
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) sugarDynCastIfPresent(From val) {
  if (!val)
    return To();
  return sugarDynCast<To>(val);
}

//===----------------------------------------------------------------------===//
// Type Identity Wrappers
//===----------------------------------------------------------------------===//

/// Peel off transparent wrappers that do not change type identity: rebind,
/// upcast, downcast, and extension.
TypedAttr stripIdentityWrappers(TypedAttr attr);

//===----------------------------------------------------------------------===//
// FuncSymbolGenerator
//===----------------------------------------------------------------------===//

/// A view of a `GeneratorAttr` whose body is a `FuncSymbolAttr`.
class FuncSymbolGenerator {
public:
  FuncSymbolGenerator() = default;
  /// Wraps `attr`, which must be null or a `GeneratorAttr` wrapping a
  /// `FuncSymbolAttr`.
  /*implicit*/ FuncSymbolGenerator(TypedAttr attr);

  /// Returns a generator over `inputParamTypes` whose body is `funcSymbol`.
  static FuncSymbolGenerator get(ArrayRef<Type> inputParamTypes,
                                 FuncSymbolAttr funcSymbol,
                                 Attribute metadata = {});

  /// Returns a fully bound generator referencing the non-parametric function
  /// `symbol` of type `type`.
  static FuncSymbolGenerator get(SymbolRefAttr symbol, FuncType type,
                                 Attribute metadata = {});

  /// Returns a fully bound generator referencing `func`.
  static FuncSymbolGenerator get(FuncOp func);

  /// Returns a generator of type `type` referencing `symbol` with
  /// `paramValues` bound. An `UnboundAttr` value, or a parameter left off the
  /// end, binds to the generator's next input parameter.
  static FuncSymbolGenerator get(SymbolRefAttr symbol,
                                 FuncTypeGeneratorType type,
                                 ArrayRef<TypedAttr> paramValues = {});
  static FuncSymbolGenerator get(StringAttr name, FuncTypeGeneratorType type,
                                 ArrayRef<TypedAttr> paramValues = {});

  /// Wraps `attr` if it is a `GeneratorAttr` wrapping a `FuncSymbolAttr`.
  static FuncSymbolGenerator dynCast(Attribute attr);

  /*implicit*/ operator Attribute() const { return storage; }
  /*implicit*/ operator TypedAttr() const { return storage; }
  /*implicit*/ operator GeneratorAttr() const { return storage; }
  explicit operator bool() const { return static_cast<bool>(storage); }

  FuncTypeGeneratorType getType() const;
  ArrayRef<Type> getInputParamTypes() const;
  bool isFullyBound();

  /// Returns the wrapped `FuncSymbolAttr`. Its parameter values are inside the
  /// generator, one scope deeper than the generator itself.
  FuncSymbolAttr getFuncSymbol() const;
  SymbolRefAttr getSymbol() const;

  /// Returns the wrapped `FuncSymbolAttr` as seen from the scope enclosing the
  /// generator. The generator must be fully bound.
  FuncType getInstantiatedType();
  FuncSymbolAttr getInstantiatedFuncSymbol();
  ArrayRef<TypedAttr> getParamValues();

private:
  GeneratorAttr storage;
};

} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// PointerLikeTypeTraits
//===----------------------------------------------------------------------===//

namespace llvm {
template <>
struct PointerLikeTypeTraits<M::KGEN::ParamDeclRefAttr>
    : public PointerLikeTypeTraits<mlir::Attribute> {
  static inline M::KGEN::ParamDeclRefAttr getFromVoidPointer(void *p) {
    return M::KGEN::ParamDeclRefAttr::getFromOpaquePointer(p);
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
/// Emit an MLIR operation call in a parameter context.
TypedAttr emitMLIROperationCall(
    StringRef opName,
    ArrayRef<std::pair<StringAttr (*)(mlir::OperationName), Attribute>> attrs,
    ArrayRef<TypedAttr> operands, Type resultType);

/// Unwrap a type reference to get to the underlying TypeGeneratorRefAttr or
/// TypeInstanceRefAttr. Types passed through generic parameters are wrapped in
/// TypeParamAttr, and this helper handles that unwrapping.
/// Returns a null TypedAttr if the type reference cannot be resolved.
TypedAttr getTypeRefForTypeValueIfResolved(TypedAttr typeRef);
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENATTRS_H
