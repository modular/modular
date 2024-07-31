//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/DomainAwareReplacer.h"
#include "mlir/IR/Operation.h"

using namespace M;

DomainAwareReplacer::DomainAwareReplacer()
    : cache([&](CacheKey attr) { return breakCycleImpl(attr); }) {}

//===----------------------------------------------------------------------===//
// Application
//===----------------------------------------------------------------------===//

void DomainAwareReplacer::replaceElementsIn(Operation *op, DomainId domain,
                                            bool replaceAttrs, bool replaceLocs,
                                            bool replaceTypes) {
  // Functor that replaces the given element if the new value is different,
  // otherwise returns nullptr.
  auto replaceIfDifferent = [&](auto element) {
    auto replacement = replace(element, domain);
    return (replacement && replacement != element) ? replacement : nullptr;
  };

  // Update the attribute dictionary.
  if (replaceAttrs) {
    if (auto newAttrs = replaceIfDifferent(op->getAttrDictionary()))
      op->setAttrs(cast<DictionaryAttr>(newAttrs));
  }

  // If we aren't updating locations or types, we're done.
  if (!replaceTypes && !replaceLocs)
    return;

  // Update the location.
  if (replaceLocs) {
    if (Attribute newLoc = replaceIfDifferent(op->getLoc()))
      op->setLoc(cast<LocationAttr>(newLoc));
  }

  // Update the result types.
  if (replaceTypes) {
    for (OpResult result : op->getResults())
      if (Type newType = replaceIfDifferent(result.getType()))
        result.setType(newType);
  }

  // Update any nested block arguments.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument &arg : block.getArguments()) {
        if (replaceLocs) {
          if (Attribute newLoc = replaceIfDifferent(arg.getLoc()))
            arg.setLoc(cast<LocationAttr>(newLoc));
        }

        if (replaceTypes) {
          if (Type newType = replaceIfDifferent(arg.getType()))
            arg.setType(newType);
        }
      }
    }
  }
}

template <typename T>
static void updateSubElementImpl(T element,
                                 DomainAwareReplacer::DomainId domain,
                                 DomainAwareReplacer &replacer,
                                 SmallVectorImpl<T> &newElements,
                                 FailureOr<bool> &changed) {
  // Bail early if we failed at any point.
  if (failed(changed))
    return;

  // Guard against potentially null inputs. We always map null to null.
  if (!element) {
    newElements.push_back(nullptr);
    return;
  }

  // Replace the element.
  if (T result = replacer.replace(element, domain)) {
    newElements.push_back(result);
    if (result != element)
      changed = true;
  } else {
    changed = failure();
  }
}

template <typename T>
static T replaceSubElements(T interface, DomainAwareReplacer::DomainId domain,
                            DomainAwareReplacer &replacer) {
  // Walk the current sub-elements, replacing them as necessary.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  FailureOr<bool> changed = false;
  interface.walkImmediateSubElements(
      [&](Attribute element) {
        updateSubElementImpl(element, domain, replacer, newAttrs, changed);
      },
      [&](Type element) {
        updateSubElementImpl(element, domain, replacer, newTypes, changed);
      });
  if (failed(changed))
    return nullptr;

  // If any sub-elements changed, use the new elements during the replacement.
  T result = interface;
  if (*changed)
    result = interface.replaceImmediateSubElements(newAttrs, newTypes);
  return result;
}

/// Shared implementation of replacing a given attribute or type element.
template <typename T, typename ReplaceFns>
static T replaceElementImpl(T element, DomainAwareReplacer::DomainId domain,
                            ReplaceFns &replaceFns,
                            DomainAwareReplacer &replacer) {
  T result = element;
  WalkResult walkResult = WalkResult::advance();
  for (auto &replaceFn : llvm::reverse(replaceFns)) {
    if (std::optional<std::pair<T, WalkResult>> newRes = replaceFn(element)) {
      std::tie(result, walkResult) = *newRes;
      break;
    }
  }

  // If an error occurred, return nullptr to indicate failure.
  if (walkResult.wasInterrupted() || !result) {
    return nullptr;
  }

  // Handle replacing sub-elements if this element is also a container.
  if (!walkResult.wasSkipped()) {
    // Replace the sub elements of this element, bailing if we fail.
    if (!(result = replaceSubElements(result, domain, replacer))) {
      return nullptr;
    }
  }

  return result;
}

template <typename T>
T DomainAwareReplacer::cachedReplaceImpl(T element, DomainId domain) {
  AttrOrType taggedElement(element);
  decltype(cache)::CacheEntry cacheEntry =
      cache.lookupOrInit(CacheKey{taggedElement, domain});
  if (auto resultOpt = cacheEntry.get())
    return T::getFromOpaquePointer(*resultOpt);

  T result;
  if constexpr (std::is_same_v<T, Attribute>)
    result =
        replaceElementImpl(element, domain, attrReplacementFns[domain], *this);
  else
    result =
        replaceElementImpl(element, domain, typeReplacementFns[domain], *this);

  cacheEntry.resolve(result.getAsOpaquePointer());
  return result;
}

std::optional<const void *>
DomainAwareReplacer::breakCycleImpl(CacheKey element) {
  AttrOrType taggedElement = element.first;
  DomainId domain = element.second;
  if (auto attr = dyn_cast<Attribute>(taggedElement)) {
    for (auto &cyclicReplaceFn : llvm::reverse(attrCycleBreakerFns[domain])) {
      if (std::optional<Attribute> newRes = cyclicReplaceFn(attr)) {
        return newRes->getAsOpaquePointer();
      }
    }
  } else {
    auto type = dyn_cast<Type>(taggedElement);
    for (auto &cyclicReplaceFn : llvm::reverse(typeCycleBreakerFns[domain])) {
      if (std::optional<Type> newRes = cyclicReplaceFn(type)) {
        return newRes->getAsOpaquePointer();
      }
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void DomainAwareReplacer::addReplacement(ReplaceFn<Attribute> fn,
                                         DomainId domain) {
  attrReplacementFns[domain].emplace_back(std::move(fn));
}

void DomainAwareReplacer::addReplacement(ReplaceFn<Type> fn, DomainId domain) {
  typeReplacementFns[domain].push_back(std::move(fn));
}

void DomainAwareReplacer::addCycleBreaker(CycleBreakerFn<Attribute> fn,
                                          DomainId domain) {
  attrCycleBreakerFns[domain].emplace_back(std::move(fn));
}

void DomainAwareReplacer::addCycleBreaker(CycleBreakerFn<Type> fn,
                                          DomainId domain) {
  typeCycleBreakerFns[domain].emplace_back(std::move(fn));
}
