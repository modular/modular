//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "LLCL/CompilerSupport/AsyncSideEffectMap.h"

namespace M::KGEN {
class FuncOp;

/// This error class is a complex error consisting of various possible nested
/// causes at certain IR locations. The error consists of a single top-level
/// simple error and potential nested errors.
class ErrorTree {
public:
  /// Construct an error tree with just a main error.
  template <typename U>
  ErrorTree(Location loc, U &&error)
      : loc(loc), error(std::forward<U>(error)) {}

  /// Construct an error tree with a main error and a nested tree of causes.
  ErrorTree(Location loc, Error error, ErrorTree causes);

  /// Construct an error with causes.
  ErrorTree(Location loc, Error error, MutableArrayRef<ErrorTree> causes);

  /// Get the location of the error.
  Location getLoc() const { return loc; }

  /// Get the main error.
  const Error &getError() const { return error; }

  /// Take the main error.
  Error takeError() { return std::move(error); }

  /// Get the causes of the main error.
  ArrayRef<ErrorTree> getCauses() const { return causes; }

  /// Get the main error message.
  StringRef getMessage() const { return error.get(); }

  /// Add a cause to the error. Return a reference to the current error tree.
  ErrorTree &addCause(ErrorTree cause) {
    causes.push_back(std::move(cause));
    return *this;
  }

  /// Add a cause to the error. Return a reference to the current error tree.
  ErrorTree &addCause(Location loc, Error cause) {
    causes.emplace_back(loc, std::move(cause));
    return *this;
  }

  /// Add a collection of causes to the error. Return a reference to the current
  /// error tree.
  ErrorTree &addCauses(MutableArrayRef<ErrorTree> causes) {
    for (ErrorTree &cause : causes)
      this->causes.push_back(std::move(cause));
    return *this;
  }

  /// Check if this error is equal to another error in contents.
  bool operator==(const ErrorTree &other) const {
    return loc == other.loc && getMessage() == other.getMessage() &&
           llvm::makeArrayRef(causes) == llvm::makeArrayRef(other.causes);
  }

  /// Explicitly copy this error.
  ErrorTree copy() const;

  /// Emit this error to an MLIR diagnostic. The main error is emitted as a
  /// diagnostic error. Any causes are emitted as notes.
  void emit(function_ref<InFlightDiagnostic(Location)> emitError) const;

private:
  /// Emit nested errors to an MLIR diagnostic as notes.
  static void emit(InFlightDiagnostic &diag, ArrayRef<ErrorTree> errors,
                   unsigned indentDepth);

  /// The location of the main error.
  Location loc;

  /// The top-level error.
  Error error;

  /// The nested causes of the main error.
  std::vector<ErrorTree> causes;
};

/// This class represents an error tree or a value.
template <typename T>
class ErrorTreeOr {
public:
  /// Create an error value.
  ErrorTreeOr(ErrorTree &&error) : value(std::move(error)) {}

  /// Create a value.
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  ErrorTreeOr(U &&value) : value(T(std::forward<U>(value))) {}

  /// Returns true if there is an error.
  bool isError() const { return std::holds_alternative<ErrorTree>(value); }

  /// Get a reference to the error, assuming there is one.
  const ErrorTree &getError() const { return std::get<ErrorTree>(value); }

  /// Take the underlying error, assuming there is one.
  ErrorTree takeError() { return std::move(std::get<ErrorTree>(value)); }

  /// Returns true if there is a valid value.
  bool hasValue() const { return std::holds_alternative<T>(value); }

  /// Get a reference to the value, assuming there is one.
  const T &getValue() const { return std::get<T>(value); }

  /// Take the underlying value, assuming there is one.
  T takeValue() { return std::move(std::get<T>(value)); }

  /// Allow implicit conversion to bool. Returns true if there is a valid value.
  operator bool() const { return hasValue(); }

  /// Allow the dereference operator to access the underlying value.
  const T &operator*() const { return getValue(); }

  /// Allow the arrow operator to access the underlying value.
  const T *operator->() const { return &getValue(); }

  /// Try to get a valid value. This method requires `T` to be
  /// default-constructible.
  T tryGetValue() const { return hasValue() ? getValue() : T(); }

  /// Explicitly copy this error or value. The value must have a copy
  /// constructor.
  ErrorTreeOr<T> copy() const {
    if (isError())
      return getError().copy();
    return getValue();
  }

private:
  /// The underlying value of this type is a variant.
  /// TODO(5864): Use a more efficient variant type when avaiable.
  std::variant<ErrorTree, T> value;
};

/// This IR evaluator is a parameter evaluator that can work during elaboration
/// to concretize parameter expressions and compute symbolic parameter
/// expressions, such as `apply` on a symbol constant or `get_sizeof` and
/// `get_alignof` a decl type.
class IREvaluator : public ParameterEvaluator {
public:
  /// Construct the IR evaluator with a symbol table for evaluating symbolic
  /// expressions.
  IREvaluator(
      SymbolTable &symtab, LLCL::AsyncSideEffectMap &asyncMap,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      DenseMap<StringAttr, Attribute> paramValues =
          DenseMap<StringAttr, Attribute>())
      : ParameterEvaluator(std::move(paramValues)), symtab(symtab),
        asyncMap(asyncMap), regionCache(std::move(regionCache)),
        transformCache(std::move(transformCache)) {}

  IREvaluator(const IREvaluator &eval)
      : ParameterEvaluator(eval), symtab(eval.symtab), asyncMap(eval.asyncMap),
        regionCache(eval.regionCache.copy()),
        transformCache(eval.transformCache.copy()) {}

  IREvaluator &operator=(const IREvaluator &eval) {
    return *new (this) IREvaluator(eval);
  }

  /// Evaluate symbolic expressions using the symbol table.
  FailureOr<TypedAttr>
  evaluateSymbolicExpression(ParamOperatorAttr op) override;

  /// Given a generic parameter expression, substitute known values for
  /// parameters into it and fold it down to a simple constant. This returns an
  /// error if a simple constant cannot be produced (e.g. because there is some
  /// dependence on target information that isn't available). If `allowUnknown`
  /// is set, only unevaluated parameter operators are rejected.
  ErrorTreeOr<Attribute> concretizeParameterExpr(Location loc, Attribute expr,
                                                 bool allowUnknown = false);
  ErrorTreeOr<Type> concretizeParameterExpr(Location loc, Type expr);

private:
  Attribute getReboundAttribute(Attribute attr) {
    return ParameterEvaluator::getReboundAttribute(attr);
  }
  Type getReboundType(Type type) {
    return ParameterEvaluator::getReboundType(type);
  }

  /// Evaluate the function with the provided constant inputs.
  ErrorTreeOr<TypedAttr> evaluateFunction(FuncOp func,
                                          ArrayRef<TypedAttr> inputs);

  /// The symbol table to lookup symbol references.
  SymbolTable &symtab;

  /// The async map to use for inflating ops.
  LLCL::AsyncSideEffectMap &asyncMap;

  /// The region cache to use for inflating ops.
  LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache;

  /// The transform cache to use for caching interpretation.
  LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;

  /// The contextual location of an error.
  Optional<Location> errorLoc;
  /// The function to use to emit an error.
  std::function<void(ErrorTree)> emitError;
};

//===----------------------------------------------------------------------===//
// evaluateConstraints implementation.
//===----------------------------------------------------------------------===//

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues. If the constraints are met, return
/// success, otherwise return why they aren't.
Optional<ErrorTree> evaluateConstraints(ArrayRef<ConstraintAttr> constraints,
                                        IREvaluator &evaluator);

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues. If the constraints are met, return
/// success, otherwise return why they aren't.
Optional<ErrorTree> evaluateConstraints(DeclInterface decl,
                                        ArrayRef<Attribute> inputParamValues,
                                        IREvaluator &evaluator);

} // namespace M::KGEN
