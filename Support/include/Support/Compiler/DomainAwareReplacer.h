//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_DOMAINAWAREREPLACER_H
#define SUPPORT_COMPILER_DOMAINAWAREREPLACER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"

namespace M {

/// A CyclicAttrTypeReplacer that allows different replacing logic under
/// different domains. A domain is a user-defined integer.
///
/// When registering a replacer, a domain can be associated with it so that
/// the replacer is only triggered under that domain. When invoking "replace"
/// on an attr/type, a domain is specified.
///
/// Replacer cache is predicated on the domain to ensure correctness.
class DomainAwareReplacer {
public:
  using DomainId = size_t;

  DomainAwareReplacer();

  //===--------------------------------------------------------------------===//
  // Application
  //===--------------------------------------------------------------------===//

  Attribute replace(Attribute element, DomainId domain) {
    return cachedReplaceImpl(element, domain);
  }

  Type replace(Type element, DomainId domain) {
    return cachedReplaceImpl(element, domain);
  }

  void replaceElementsIn(Operation *op, DomainId domain,
                         bool replaceAttrs = true, bool replaceLocs = false,
                         bool replaceTypes = false);

  //===--------------------------------------------------------------------===//
  // Registration - Replacers
  //===--------------------------------------------------------------------===//

  template <typename T>
  using ReplaceFnResult = std::optional<std::pair<T, WalkResult>>;
  template <typename T>
  using ReplaceFn = std::function<ReplaceFnResult<T>(T)>;

  void addReplacement(ReplaceFn<Attribute> fn, DomainId domain);
  void addReplacement(ReplaceFn<Type> fn, DomainId domain);

  /// Register a replacement function that doesn't match the default signature,
  /// either because it uses a derived parameter type, or it uses a simplified
  /// result type.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>,
            typename ResultT = std::invoke_result_t<FnT, T>>
  std::enable_if_t<!std::is_same_v<T, BaseT> ||
                   !std::is_convertible_v<ResultT, ReplaceFnResult<BaseT>>>
  addReplacement(FnT &&callback, DomainId domain) {
    addReplacement(
        [callback = std::forward<FnT>(callback)](
            BaseT base) -> ReplaceFnResult<BaseT> {
          if (auto derived = dyn_cast<T>(base)) {
            if constexpr (std::is_convertible_v<ResultT,
                                                std::optional<BaseT>>) {
              std::optional<BaseT> result = callback(derived);
              return result ? std::make_pair(*result, WalkResult::advance())
                            : ReplaceFnResult<BaseT>();
            } else {
              return callback(derived);
            }
          }
          return ReplaceFnResult<BaseT>();
        },
        domain);
  }

  //===--------------------------------------------------------------------===//
  // Registration - CycleBreakers
  //===--------------------------------------------------------------------===//

  template <typename T>
  using CycleBreakerFn = std::function<std::optional<T>(T)>;

  void addCycleBreaker(CycleBreakerFn<Attribute> fn, DomainId domain);
  void addCycleBreaker(CycleBreakerFn<Type> fn, DomainId domain);

  /// Register a cycle-breaking function that doesn't match the default
  /// signature.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>>
  std::enable_if_t<!std::is_same_v<T, BaseT>> addCycleBreaker(FnT &&callback,
                                                              DomainId domain) {
    addCycleBreaker(
        [callback =
             std::forward<FnT>(callback)](BaseT base) -> std::optional<BaseT> {
          if (auto derived = dyn_cast<T>(base))
            return callback(derived);
          return std::nullopt;
        },
        domain);
  }

private:
  using AttrOrType = PointerUnion<Attribute, Type>;
  using CacheKey = std::pair<AttrOrType, DomainId>;

  /// Invokes the registered cycle-breaker functions from most recently
  /// registered to least recently registered until a successful result is
  /// returned.
  std::optional<const void *> breakCycleImpl(CacheKey element);

  /// Shared concrete implementation of the public `replace` functions.
  template <typename T>
  T cachedReplaceImpl(T element, DomainId domain);

  /// The set of replacement functions that map sub elements.
  DenseMap<DomainId, SmallVector<ReplaceFn<Attribute>>> attrReplacementFns;
  DenseMap<DomainId, SmallVector<ReplaceFn<Type>>> typeReplacementFns;

  /// The set of registered cycle-breaker functions.
  DenseMap<DomainId, SmallVector<CycleBreakerFn<Attribute>>>
      attrCycleBreakerFns;
  DenseMap<DomainId, SmallVector<CycleBreakerFn<Type>>> typeCycleBreakerFns;

  mlir::CyclicReplacerCache<CacheKey, const void *> cache;
};

} // namespace M

#endif // SUPPORT_COMPILER_DOMAINAWAREREPLACER_H
