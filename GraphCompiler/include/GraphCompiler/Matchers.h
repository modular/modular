//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file provides MO/MOGG specific Pattern matching intended to be
// used when lowering MO ops to more complex kernels or performing
// simplifications
//
// This is not a final solution, longer term we would like this infrastructure
// to live in MOJO, so this is intended to be the lightest weight solution
// until such a time that mojo is capable of expressing this kind of
// functionality
//
// The code in this file is forked and heavily modified from mlir/IR/Matchers.h
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MATCHERS_H
#define GENERICML_GRAPHCOMPILER_MATCHERS_H

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "Support/AssertStream.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace GraphCompiler {

template <typename UnnamedType>
std::string stringifyMLIRObject(UnnamedType &&obj) {
  mlir::OpPrintingFlags flg;
  std::string strRepr;
  llvm::raw_string_ostream stream(strRepr);
  obj.print(stream, flg.useLocalScope());

  return strRepr;
}
// Debug log of pattern matching; keeps only the nodes visited on the way to a
// failure.
class Logger {
public:
  template <typename T>
  void push(T &&val) {
    stack.push_back(stringifyMLIRObject(val));
  }
  // Pop back to a fork once it succeeded; successful forks aren't interesting.
  template <typename T>
  void pop(T &&val) {
    auto string = stringifyMLIRObject(val);
    // Drop what was pushed after this op, e.g. a failed optional branch.
    auto pos = std::find(stack.rbegin(), stack.rend(), string);
    stack.erase(stack.end() - std::distance(stack.rbegin(), pos), stack.end());
  }
  void markDeadEnd(const std::string &pattern) {
    stack.emplace_back("Hit a Dead End at " + pattern + "! Rewinding Stack.");
  }
  void print(llvm::raw_ostream &os) {
    tabs = "";
    for (auto i = stack.rbegin(), e = stack.rend(); i != e; ++i) {
      if ((*i).rfind("Hit a Dead End", 0) == 0)
        tabs = "";
      os << tabs + *i << '\n';
      tabs += " ";
    }
  }

private:
  std::vector<std::string> stack;
  std::string tabs = "";
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Logger &log) {
  log.print(os);
  return os;
}

namespace Detail {

/// Terminal matcher, always returns true.
struct AnyValueMatcher {
  bool match(mlir::Value op, Logger *log = nullptr) const { return true; }
};
/// Trait to check whether T provides a 'match' method with type
/// `MatchTarget` (Value, Operation, or Attribute).
template <typename T, typename MatchTarget>
using has_compatible_matcher_t = decltype(std::declval<T>().match(
    std::declval<MatchTarget>(), std::declval<Logger *>()));

/// Statically switch to a Value matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<Detail::has_compatible_matcher_t,
                                   MatcherClass, mlir::Value>::value,
                 bool>
matchOperandOrValue(mlir::Value op, MatcherClass &matcher,
                    Logger *log = nullptr) {
  return matcher.match(op, log);
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<Detail::has_compatible_matcher_t,
                                   MatcherClass, mlir::Operation *>::value,
                 bool>
matchOperandOrValue(mlir::Value op, MatcherClass &matcher,
                    Logger *log = nullptr) {
  if (auto *defOp = op.getDefiningOp())
    return matcher.match(defOp, log);
  return false;
}

/// Statically switch to a Value matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<Detail::has_compatible_matcher_t,
                                   MatcherClass, mlir::Value>::value,
                 bool>
matchOperandOrValueAtIndex(mlir::Operation *op, unsigned idx,
                           MatcherClass &matcher, Logger *log = nullptr) {
  return matcher.match(op->getOperand(idx), log);
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
std::enable_if_t<llvm::is_detected<Detail::has_compatible_matcher_t,
                                   MatcherClass, mlir::Operation *>::value,
                 bool>
matchOperandOrValueAtIndex(mlir::Operation *op, unsigned idx,
                           MatcherClass &matcher, Logger *log = nullptr) {
  if (auto *defOp = op->getOperand(idx).getDefiningOp())
    return matcher.match(defOp, log);
  return false;
}

template <typename TupleT, class CallbackT, std::size_t... Is>
constexpr void enumerateImpl(TupleT &&tuple, CallbackT &&callback,
                             std::index_sequence<Is...>) {

  (callback(std::integral_constant<std::size_t, Is>{}, std::get<Is>(tuple)),
   ...);
}

template <typename... Tys, typename CallbackT>
constexpr void enumerate(std::tuple<Tys...> &tuple, CallbackT &&callback) {
  Detail::enumerateImpl(tuple, std::forward<CallbackT>(callback),
                        std::make_index_sequence<sizeof...(Tys)>{});
}

/// RecursivePatternMatcher that composes. Modified from upstream to support
/// logging, with SFINAE to get pattern op names.
template <typename OpType, typename... OperandMatchers>
struct RecursivePatternMatcher {
  RecursivePatternMatcher(OperandMatchers... matchers)
      : operandMatchers(matchers...) {}
  using yes = char[1];
  using no = char[2];
  bool match(mlir::Operation *op, Logger *log = nullptr) {
    if (log)
      log->push(*op);
    if (!llvm::isa<OpType>(op)) {
      if (log)
        log->markDeadEnd(std::string("OpPattern " + getName() +
                                     " didn't match MLIR OpType"));
      return false;
    }
    if (op->getNumOperands() != sizeof...(OperandMatchers)) {
      if (log)
        log->markDeadEnd(std::string(
            "OpPattern \"" + getName() + "\" had " +
            std::to_string(sizeof...(OperandMatchers)) +
            " while the mlir op had " + std::to_string(op->getNumOperands())));
      return false;
    }
    bool res = true;
    enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, index, matcher, log);
    });
    if (log && res)
      log->pop(*op);
    return res;
  }

  template <typename T>
  using has_name_t = decltype(std::declval<T>().getOperationName());

  template <class Q = OpType>
  typename std::enable_if<llvm::is_detected<has_name_t, Q>::value,
                          std::string>::type
  getName() {
    return Q::getOperationName().str();
  }

  template <class Q = OpType>
  typename std::enable_if<!llvm::is_detected<has_name_t, Q>::value,
                          std::string>::type
  getName() {
    return "Interface Op";
  }

  std::tuple<OperandMatchers...> operandMatchers;
};

/// RecursivePatternMatcher that composes.
template <typename... OperandMatchers>
struct UnregisteredRecursivePatternMatcher {
  UnregisteredRecursivePatternMatcher(mlir::StringRef opName,
                                      OperandMatchers... matchers)
      : opName(opName), operandMatchers(matchers...) {}
  bool match(mlir::Operation *op, Logger *log = nullptr) {
    if (log)
      log->push(*op);
    if (op->getName().getStringRef() != opName) {
      if (log)
        log->markDeadEnd(std::string("OpPattern " + opName.str() +
                                     " didn't match MLIR OpType"));
      return false;
    }
    if (op->getNumOperands() != sizeof...(OperandMatchers)) {
      if (log)
        log->markDeadEnd(std::string(
            "OpPattern " + opName.str() + " had " +
            std::to_string(sizeof...(OperandMatchers)) +
            " while the mlir op had " + std::to_string(op->getNumOperands())));
      return false;
    }
    bool res = true;
    enumerate(operandMatchers, [&](size_t idx, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, idx, matcher, log);
    });

    if (log && res)
      log->pop(*op);

    return res;
  }

private:
  llvm::StringRef opName;
  std::tuple<OperandMatchers...> operandMatchers;
};

template <typename T, typename Matcher>
struct Capture {
  Capture(T *bindDest, Matcher matcher)
      : bindDest(*bindDest), matcher(matcher) {}
  bool match(T op, Logger *log = nullptr) {
    if (!matcher.match(op, log))
      return false;
    // Already logged by the match.
    bindDest = op;
    return true;
  }

private:
  T &bindDest;
  Matcher matcher;
};

// CTAD Declaration to make the compiler happy
template <typename T, typename Matcher>
Capture(T, Matcher) -> Capture<T, Matcher>;

template <typename U, typename Matcher>
struct StringAttrMatcher {
  StringAttrMatcher(llvm::StringRef name, U value, Matcher matcher)
      : name(name), value(value), matcher(matcher) {}
  bool match(mlir::Operation *op, Logger *log = nullptr) {
    if (log)
      log->push(*op);
    bool res =
        op->hasAttr(name) && op->getAttrOfType<mlir::StringAttr>(name) == value;
    if (!res && log)
      log->markDeadEnd("StringAttrPattern");
    res &= matcher.match(op, log);
    if (log && res)
      log->pop(*op);
    return res;
  }

private:
  llvm::StringRef name;
  U value;
  Matcher matcher;
};

// CTAD Declaration to make the compiler happy
template <typename U, typename Matcher>
StringAttrMatcher(U, Matcher) -> StringAttrMatcher<U, Matcher>;

// Find a scalar constant that matches a particular variable.
template <typename T>
struct MOScalarConstantMatcher {
  MOScalarConstantMatcher(T targetValue) : targetValue(targetValue) {}

  bool match(mlir::Value val, Logger *log = nullptr) {
    if (log)
      log->push(val);
    auto tensorVal = llvm::dyn_cast<M::MO::TensorValue>(val);
    if (!tensorVal)
      return false;

    std::optional<llvm::SmallVector<T>> maybeInt =
        M::MO::tryGetConstantOfRank<T>(tensorVal, 0);
    // TODO: Make this approximate for floating point numbers?
    bool res = false;
    if (maybeInt.has_value()) {
      res = maybeInt.value().front() == targetValue;
      if (log && !res)
        log->markDeadEnd("MOScalarConstant expected" +
                         std::to_string(targetValue));
    } else {
      if (log)
        log->markDeadEnd("MOScalarConstant did not find a constant");
    }
    if (log && res)
      log->pop(val);
    return res;
  }

private:
  T targetValue;
};

// CTAD Declaration to make the compiler happy
template <typename T>
MOScalarConstantMatcher(T) -> MOScalarConstantMatcher<T>;

// Find a splat constant that matches a particular variable.
template <typename T, bool typed>
struct MOSplatConstantMatcher {
  MOSplatConstantMatcher(T targetValue) : targetValue(targetValue) {}

  static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                "Unsupported type");

  bool match(mlir::Value val, Logger *log = nullptr) {
    if (log)
      log->push(val);

    auto tensorVal = llvm::dyn_cast<M::MO::TensorValue>(val);
    if (!tensorVal)
      return false;

    auto tensorType = llvm::dyn_cast<M::MO::TensorType>(val.getType());
    if (!tensorType)
      return false;

    std::optional<bool> res;
    res = matchForType<float>(targetValue, tensorVal, tensorType, val, log);
    if (res.has_value())
      return *res;
    res = matchForType<double>(targetValue, tensorVal, tensorType, val, log);
    if (res.has_value())
      return *res;
    res = matchForType<int32_t>(targetValue, tensorVal, tensorType, val, log);
    if (res.has_value())
      return *res;
    res = matchForType<int64_t>(targetValue, tensorVal, tensorType, val, log);
    if (res.has_value())
      return *res;
    else if (log) {
      log->markDeadEnd("MOSplatConstantMatcher did not find a constant in the "
                       "supported types");
    }

    return false;
  }

private:
  template <typename U>
  std::optional<bool> matchForType(U targetValue, M::MO::TensorValue tensorVal,
                                   M::MO::TensorType tensorType,
                                   mlir::Value val, Logger *log = nullptr) {
    if constexpr (typed && !std::is_same_v<T, U>)
      return {};

    int rank = 0;
    if (tensorType.hasRank()) {
      rank = tensorType.getRank();
    }

    auto maybeValues = M::MO::tryGetConstantOfRank<U>(tensorVal, rank);
    if (!maybeValues.has_value())
      return std::nullopt;

    for (auto value : maybeValues.value()) {
      if (value != targetValue) {
        if (log) {
          log->markDeadEnd("MOSplatConstantMatcher expected" +
                           std::to_string(targetValue));
        }
        return false;
      }
    }

    if (log)
      log->pop(val);

    return true;
  }

  T targetValue;
};

// Check a value for a particular rank
template <typename Matcher>
struct MORankedValueMatcher {
  MORankedValueMatcher(int64_t targetRank, Matcher matcher)
      : targetRank(targetRank), matcher(matcher) {}
  bool match(mlir::Value op, Logger *log = nullptr) {
    if (log)
      log->push(op);
    bool res = false;
    if (auto type = llvm::dyn_cast<mlir::ShapedType>(op.getType())) {
      res = type.hasRank() && type.getRank() == targetRank;
      if (!res && log) {
        log->markDeadEnd("MORankedValue expected " +
                         std::to_string(targetRank));
        return false;
      }
      res &= matcher.match(op, log);
    }
    if (auto type = llvm::dyn_cast<M::MO::TensorType>(op.getType())) {
      res =
          type.hasRank() && static_cast<int64_t>(type.getRank()) == targetRank;
      if (!res && log) {
        log->markDeadEnd("MORankedValue expected " +
                         std::to_string(targetRank));
        return false;
      }
      res &= matcher.match(op, log);
    }
    if (log && res)
      log->pop(op);
    return res;
  }

private:
  int64_t targetRank;
  Matcher matcher;
};

// Check a Value for a partciular Type
template <typename T, typename Matcher>
struct TypeMatcher {
  TypeMatcher(Matcher matcher) : matcher(matcher) {}
  bool match(mlir::Value op, Logger *log = nullptr) {
    bool res = false;
    if (log)
      log->push(op);
    if (llvm::dyn_cast<T>(op.getType()))
      res = matcher.match(op, log);
    else if (log)
      log->markDeadEnd("TypeMatcher");
    if (log && res)
      log->pop(op);
    return res;
  }

private:
  Matcher matcher;
};

// Matches if either pattern matches the subgraph.
template <typename Matcher1, typename Matcher2>
struct OptionalMatcher {
  OptionalMatcher(Matcher1 matcher1, Matcher2 matcher2)
      : matcher1(matcher1), matcher2(matcher2) {}
  bool match(mlir::Value op, Logger *log = nullptr) {
    if (log)
      log->push(op);
    bool res = matchOperandOrValue(op, matcher1, log) ||
               matchOperandOrValue(op, matcher2, log);
    if (log && res)
      log->pop(op);
    return res;
  }

private:
  Matcher1 matcher1;
  Matcher2 matcher2;
};

// Cross-dialect check for ConstantLike ops.
struct constant_op_matcher {
  bool match(mlir::Operation *op, Logger *log = nullptr) {
    if (log)
      log->push(*op);
    bool res = op->hasTrait<mlir::OpTrait::ConstantLike>();
    if (log && res)
      log->pop(*op);
    else if (log)
      log->markDeadEnd("ConstantOpMatcher");
    return res;
  }
};

/// The matcher that matches operations that have the `ConstantLike` trait, and
/// binds the folded attribute value.
template <typename AttrT>
struct constant_op_binder {
  AttrT *bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT *bind_value) : bind_value(bind_value) {}
  /// Creates a matcher instance that doesn't bind if match succeeds.
  constant_op_binder() : bind_value(nullptr) {}

  bool match(mlir::Operation *op, Logger *log = nullptr) {
    if (log)
      log->push(*op);
    if (!op->hasTrait<mlir::OpTrait::ConstantLike>()) {
      if (log)
        log->markDeadEnd("ConstantOpBinder");
      return false;
    }

    // Fold the constant to an attribute.
    llvm::SmallVector<mlir::OpFoldResult, 1> foldedOp;
    mlir::LogicalResult result = op->fold(/*operands=*/{}, foldedOp);
    (void)result;
    ASSERT_STREAM(succeeded(result),
                  << "expected ConstantLike op to be foldable");

    if (auto attr = llvm::dyn_cast<AttrT>(foldedOp.front())) {
      if (bind_value)
        *bind_value = attr;
      if (log)
        log->pop(*op);
      else if (log)
        log->markDeadEnd("ConstantOpBinder");
      return true;
    }
    return false;
  }
};

template <typename AttrT>
inline Detail::constant_op_binder<AttrT> m_Constant(AttrT *bind_value) {
  return Detail::constant_op_binder<AttrT>(bind_value);
}

} // namespace Detail

namespace Matchers {

template <typename OpType, typename... Matchers>
auto m_Op(Matchers... matchers) {
  return Detail::RecursivePatternMatcher<OpType, Matchers...>(matchers...);
}

template <typename OpType, typename... Matchers>
auto m_Op(mlir::Operation **bindDest, Matchers... matchers) {
  return Detail::Capture(bindDest, m_Op<OpType, Matchers...>(matchers...));
}

template <typename... Matchers>
auto m_Op(llvm::StringRef opName, Matchers... matchers) {
  return Detail::UnregisteredRecursivePatternMatcher<Matchers...>(opName,
                                                                  matchers...);
}

template <typename... Matchers>
auto m_Op(mlir::Operation **bindDest, llvm::StringRef opName,
          Matchers... matchers) {
  return Detail::Capture(bindDest, m_Op(opName, matchers...));
}

template <typename Matcher>
auto m_StringAttr(llvm::StringRef attrName, llvm::StringRef opName,
                  Matcher matcher) {
  return Detail::StringAttrMatcher(attrName, opName, matcher);
}

template <typename... Matchers>
auto m_StringAttr(mlir::Operation **bindDest, llvm::StringRef attrName,
                  llvm::StringRef opName, Matchers... matchers) {
  return Detail::Capture(bindDest, m_StringAttr(attrName, opName, matchers...));
}

template <typename Matcher>
auto m_Ranked(int64_t targetRank, Matcher matcher) {
  return Detail::MORankedValueMatcher<Matcher>(targetRank, matcher);
}

template <typename T>
auto m_ScalarConstant(T targetValue) {
  return Detail::MOScalarConstantMatcher<T>(targetValue);
}

template <typename T>
auto m_ScalarConstant(mlir::Value *val, T targetValue) {
  return Detail::Capture(val, m_ScalarConstant(targetValue));
}

template <typename T>
auto m_SplatConstant(T targetValue) {
  return Detail::MOSplatConstantMatcher<T, true>(targetValue);
}

template <typename T>
auto m_SplatConstant(mlir::Value *val, T targetValue) {
  return Detail::Capture(val,
                         Detail::MOSplatConstantMatcher<T, true>(targetValue));
}

template <typename T>
auto m_SplatConstantOfAnyType(T targetValue) {
  return Detail::MOSplatConstantMatcher<T, false>(targetValue);
}

template <typename T>
auto m_SplatConstantOfAnyType(mlir::Value *val, T targetValue) {
  return Detail::Capture(val,
                         Detail::MOSplatConstantMatcher<T, false>(targetValue));
}

template <typename Matcher1, typename Matcher2>
auto m_Optional(Matcher1 matcher1, Matcher2 matcher2) {
  return Detail::OptionalMatcher<Matcher1, Matcher2>(matcher1, matcher2);
}

template <typename Matcher, typename... Matchers>
auto m_Optional(Matcher matcher, Matchers... matchers) {
  return m_Optional(matcher, m_Optional(matchers...));
}

template <typename BCastType, typename T>
auto m_SplatOrBroadcastedConstantOfAnyType(T targetValue) {
  return m_Optional(m_SplatConstantOfAnyType(targetValue),
                    m_Op<BCastType>(m_SplatConstantOfAnyType(targetValue)));
}

template <typename BCastType, typename T>
auto m_SplatOrBroadcastedConstantOfAnyType(mlir::Value *val, T targetValue) {
  return m_Optional(
      m_SplatConstantOfAnyType(val, targetValue),
      m_Op<BCastType>(m_SplatConstantOfAnyType(val, targetValue)));
}

template <typename T, typename Matcher>
auto m_Typed(Matcher matcher) {
  return Detail::TypeMatcher<T, Matcher>(matcher);
}

template <typename T, typename Matcher>
auto m_Typed(mlir::Value *val, Matcher matcher) {
  return Detail::Capture(val, m_Typed<T, Matcher>(matcher));
}

template <typename Matcher>
auto m_MOVector(Matcher matcher) {
  return m_Ranked(1, m_Typed<M::MO::TensorType, Matcher>(matcher));
}

template <typename Matcher>
auto m_MOMatrix(Matcher matcher) {
  return m_Ranked(2, m_Typed<M::MO::TensorType, Matcher>(matcher));
}

template <typename OpType, typename Matcher1, typename Matcher2>
auto m_BinaryCommutativeOp(Matcher1 matcher1, Matcher2 matcher2) {
  return m_Optional(m_Op<OpType>(matcher1, matcher2),
                    m_Op<OpType>(matcher2, matcher1));
}

template <typename OpType, typename Matcher1, typename Matcher2>
auto m_BinaryCommutativeOp(mlir::Operation **bindDest, Matcher1 matcher1,
                           Matcher2 matcher2) {
  return m_Optional(m_Op<OpType>(bindDest, matcher1, matcher2),
                    m_Op<OpType>(bindDest, matcher2, matcher1));
}

inline auto m_Any() { return Detail::AnyValueMatcher(); }
inline auto m_Any(mlir::Value *val) { return Detail::Capture(val, m_Any()); }

inline auto m_Constant() { return Detail::constant_op_matcher(); }

inline auto m_Constant(mlir::Value *val) {
  return Detail::Capture(val, m_Constant());
}

/// Matches a value from a constant foldable operation and writes the value to
/// bind_value.
template <typename AttrT>
inline Detail::constant_op_binder<AttrT> m_Constant(AttrT *bind_value) {
  return Detail::constant_op_binder<AttrT>(bind_value);
}
/// Entry point for matching a pattern over a Value.
template <typename Pattern>
inline bool matchPattern(mlir::Value value, const Pattern &pattern) {
  assert(value);
  // TODO: handle other cases
  if (auto *op = value.getDefiningOp())
    return const_cast<Pattern &>(pattern).match(op);
  return false;
}

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool matchPattern(mlir::Operation *op, const Pattern &pattern) {
  assert(op);
  return const_cast<Pattern &>(pattern).match(op);
}

} // namespace Matchers
} // namespace GraphCompiler

namespace M {
namespace MO {

/// Returns a parameter attribute through a variant of ToTensor ops if possible.
/// Otherwise returns a null attribute.
template <typename ToTensorOpTy>
static TypedAttr getParamAttrThrough(TensorValue inputVal) {
  if (auto toTensorOp = inputVal.getDefiningOp<ToTensorOpTy>()) {
    if (auto paramToValueOp =
            toTensorOp.getInput()
                .template getDefiningOp<MOSH::ParamToValueOp>()) {
      return paramToValueOp.getValue();
    }
  }
  return {};
}

} // namespace MO
} // namespace M
#endif // GENERICML_GRAPHCOMPILER_MATCHERS_H
